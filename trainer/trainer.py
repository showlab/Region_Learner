import numpy as np
import torch
from torch import nn
from base.base_trainer import Multi_BaseTrainer_dist, BaseTrainer
from utils import inf_loop
from model.model import sim_matrix
import torch.distributed as dist
from torch import nn
import time

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(n_gpu)]
        dist.all_gather(output, tensor)
        ctx.local_rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.local_rank : ctx.batch_size * (ctx.local_rank + 1)],
            None, None,
        )

class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )



class Multi_Trainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        # modified by Mr. Yan
        # org version use a fixed lr in FiT, but now we reduce lr by schedule. 
        # But the init learning_rate should be configed by json rather than fixed. Thus we use "lr = args.learning_rate"
        # TODO
        # lr = args.learning_rate1
        # print('trainer', args)
        # lr = args.learning_rate
        # for milestone in args.schedule:
        #     lr *= 0.1 if epoch >= milestone else 1.
        for i, param_group in enumerate(optimizer.param_groups):
            # print(param_group['lr'])
            lr = param_group['lr']
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
            param_group['lr'] = lr
            # print(type(param_group['params']))
            # print(type(param_group['params'][0]))
            # print(type(param_group['params'][0]))
            if args.local_rank == 0:
                print('current learning rate is:\t', param_group['lr'])

    def _train_epoch(self, epoch, scaler):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        
        # added by Mr Yan
        since = time.time()
        # BUG fixed, should put lr adjust before the first epoch
        self._adjust_learning_rate(self.optimizer, epoch, self.args)
        
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            # Dist mode puts a batch_size of samples on each gpu, rather than spliting a batch_size of data onto all gpus.
            # so, here need to *self.n_gpu
            if (batch_idx + 1) * self.total_batch_sum*self.n_gpu > self.max_samples_per_epoch:
                break
            # if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
            #     break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)

                # self.optimizer.zero_grad() # removed by Mr. YAN
                # Runs the forward pass under autocast.
                with torch.cuda.amp.autocast(enabled=self.use_amp): # Automatic Mixed Precision added by Mr. Yan
                    with torch.set_grad_enabled(True):
                        if 'QA' in self.data_loader[dl_idx].dataset_name:
                            # target = torch.LongTensor(data['answer_id']).to(self.device)
                            
                            target = data['answer_id'].to(self.device)
                            # treat QA as a classification problem
                            output = self.model(data, epoch)
                            loss = self.loss(output, target)
                        else:
                            text_embeds, video_embeds = self.model(data, epoch)
                            video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                            text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                            # loss = retrival_head(text_embeds, video_embeds)
                            output = sim_matrix(text_embeds, video_embeds)
                            loss = self.loss(output)

                        # normalize loss to account for batch accumulation
                        loss = loss / self.accum_iter
                        
                        #loss = self.loss(output) * self.args.world_size

                ################## Automatic Mixed Precision added by Mr. Yan########################
                # loss.backward()
                # self.optimizer.step()

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()


                # weights update with Gradient Accumulation, added by Mr. YAN
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(data_li)):
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(self.optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()

                    self.optimizer.zero_grad() # put here by Mr. YAN
                ###################################################################################

                detached_loss = loss.detach().item()
                
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', detached_loss)

                total_loss[dl_idx] += detached_loss

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('[{:.2f}s] Train Epoch: {} dl{} {} Loss: {:.6f}'.format(time.time()-since,
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    # added by Mr Yan
                    since = time.time()


                # self.optimizer.zero_grad() # original position
            
            if batch_idx == self.len_epoch:
                break
            
            # added by Mr. Yan for debuging
            if self.args.debug and batch_idx>self.accum_iter: # for testing Gradient Accumulation
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        # self._adjust_learning_rate(self.optimizer, epoch, self.args) # BUG fixed

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        #     try:
        #         print('current learning rate is:\t', self.lr_scheduler.get_lr()[0])
        #     except:
        #         print('error during print lr')
        # else:
        #     try:
        #         print('fixed learning rate is:\t', args.lr)
        #     except:
        #         print('error during print lr')

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        # for zero-shot action recognition.
        # Added by Mr. YAN
        
        alltext_list = []
        answer_list = []
        answer_all_arr = {x: [] for x in range(len(self.valid_data_loader))} # for MC
        logits_all_arr = {x: [] for x in range(len(self.valid_data_loader))} # for QA
        target_all_arr = {x: [] for x in range(len(self.valid_data_loader))} # for QA
        
        res_dict = {}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    #

                    # if 'QA' in self.valid_data_loader[dl_idx].dataset_name:
                    #     answer = torch.LongTensor(data['answer_id']).cuda()
                    #     answer_all = [torch.zeros_like(answer) for _ in range(self.n_gpu)]
                    #     torch.distributed.all_gather(answer_all, answer)
                    #     answer_all = torch.cat(answer_all, dim=0)
                    #     # print('len', len(answer_all))
                    #     answer_all_arr[dl_idx].append(answer_all.cpu())
                    #     # print(data['text'][:5])

                    if self.valid_data_loader[dl_idx].dataset_name in ['HMDB', 'UCF']:
                        # label_set.update(data['text'])
                        alltext_list.extend(data['text'])

                    if self.valid_data_loader[dl_idx].dataset_name in ['MSRVTT_MC', 'LSMDC_MC']:
                        # caption is a dict containing five options
                        new_data = []
                        opt_size = len(data['text'])
                        batch_size = len(data['text'][0])
                        for b in range(batch_size):
                            for i in range(opt_size):
                                new_data.append(data['text'][i][b])
                        data['text'] = new_data
                        # answer_list.extend(data['answer'])

                        answer = torch.LongTensor(data['answer']).cuda()
                        answer_all = [torch.zeros_like(answer) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(answer_all, answer)
                        answer_all = torch.cat(answer_all, dim=0)
                        # print('len', len(answer_all))
                        answer_all_arr[dl_idx].append(answer_all.cpu())
                        # print(data['text'][:5])



                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                        data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)
                    
                    # Runs the forward pass under autocast.
                    with torch.cuda.amp.autocast(enabled=self.use_amp): # Automatic Mixed Precision added by Mr. Yan
                        if 'QA' in self.valid_data_loader[dl_idx].dataset_name:
                            target = torch.LongTensor(data['answer_id']).cuda()

                            # target = data['answer_id'].cuda()
                            target_all = [torch.zeros_like(target) for _ in range(self.n_gpu)]
                            torch.distributed.all_gather(target_all, target)
                            target_all = torch.cat(target_all, dim=0)
                            target_all_arr[dl_idx].append(target_all.cpu())


                            logits = self.model(data)
                            loss = self.loss(logits, target)

                            logits_all = [torch.zeros_like(logits) for _ in range(self.n_gpu)]
                            torch.distributed.all_gather(logits_all, logits)
                            logits_all = torch.cat(logits_all, dim=0)
                            logits_all_arr[dl_idx].append(logits_all.cpu())

                        else:
                            text_embed, vid_embed = self.model(data, return_embeds=True)
                            #if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                                # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                                # It can happen with the last batch of the dataset, depending on its size.
                                # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                            #    text_embed, vid_embed = self.model.module(data, return_embeds=True)
                            #else:
                            #    text_embed, vid_embed = self.model(data, return_embeds=True)
                            vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                            torch.distributed.all_gather(vid_embed_all, vid_embed)
                            vid_embed_all = torch.cat(vid_embed_all, dim=0)

                            text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                            torch.distributed.all_gather(text_embed_all, text_embed)
                            text_embed_all = torch.cat(text_embed_all, dim=0)


                            text_embed_arr[dl_idx].append(text_embed_all.cpu())
                            vid_embed_arr[dl_idx].append(vid_embed_all.cpu())
                            sims_batch = sim_matrix(text_embed_all, vid_embed_all)
                            loss = self.loss(sims_batch)


                        total_val_loss[dl_idx] += loss.item()
                    
                    # added by Mr. Yan for debuging
                    if self.args.debug:
                        break
                
                
   
        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None and self.args.rank == 0:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}


            # for MSRVTT/MSVD QA
            if 'QA' in self.valid_data_loader[dl_idx].dataset_name:
                logits_all = torch.cat(logits_all_arr[dl_idx])
                # ac = (out==ans).float().mean().item()
                gt = torch.cat(target_all_arr[dl_idx])
                # print('MSRVTT_QA: ', logits_all.shape, gt.shape)
                for metric in self.metrics:
                    res = metric(logits_all, gt)
                    if self.args.rank == 0:
                        print("Question Answer:", res)
                        res_dict['Question Answer'] = res
                break

                            
            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])

            # for zero-shot action recognition, we need fix the text_embeds within limited action labels.
            # added by Mr. Yan
            if self.valid_data_loader[dl_idx].dataset_name in ['HMDB', 'UCF']:
                class_id = []
                label_list = list(set(alltext_list))
                print('Number of Action Classes:', len(label_list))
                if self.tokenizer is not None:
                    label_tokens = self.tokenizer(label_list, return_tensors='pt', padding=True, truncation=True)
                    label_tokens = {key: val.to(self.device) for key, val in label_tokens.items()}
                text_embeds = self.model.module.compute_text((label_tokens)).cpu()
                # print('text_embeds size:', text_embeds.size())
                sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

                for item in alltext_list:
                    class_id.append(label_list.index(item))
                for metric in self.metrics:
                    res = metric(sims, class_id)
                    if self.args.rank == 0:
                        print("Zero-Shot Action Recognition:", res)
                        res_dict['Zero-Shot Action Recognition'] = res
                break

            # for MSRVTT/LSMDC MC
            if 'MC' in self.valid_data_loader[dl_idx].dataset_name:
                answer_all = torch.cat(answer_all_arr[dl_idx])
                B = text_embeds.size(0)
                text_embeds = text_embeds.reshape(B//5, 5, -1)
                # print('embeds:', text_embeds.size(), vid_embeds.size())
                all_sims = []
                for b in range(text_embeds.size(0)):
                    sims = sim_matrix(vid_embeds[b].unsqueeze(0), text_embeds[b])
                    # print('sims:', sims.size())
                    all_sims.append(sims)


                for metric in self.metrics:
                    # print('answer_all', len(answer_all))
                    res = metric(torch.stack(all_sims).squeeze(), answer_all)
                    if self.args.rank == 0:
                        print("Multiple Choice:", res)
                        res_dict['Multiple Choice'] = res
                break



            # print("text_embeds shape:", text_embeds.shape)
            # print("vid_embeds shape:", vid_embeds.shape)

            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()
            for metric in self.metrics:
                metric_name = metric.__name__
                # print('sims shape:', sims[:10, :10])
                res = metric(sims)
                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        
        if self.args.rank == 0:
            for dl_idx in range(len(self.valid_data_loader)):
                res_dict[f'val_loss_{dl_idx}'] = total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
            res_dict['nested_val_metrics'] = nested_metrics


        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                  truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()
                text_embeds, video_embeds = self.model(data)
                output = sim_matrix(text_embeds, video_embeds)
                loss = self.loss(output)
                loss.backward()
                #for name, param in self.model.named_parameters():
                #    if 'video_model.blocks.0.norm1.weight' in name:
                #        print(batch_idx, name, param.grad.cpu().numpy()[0:10])
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    #for name, param in self.model.named_parameters():
                    #    if 'video_model.temporal_embed' in name:
                    #        print(param[:,:,0])
                self.optimizer.zero_grad()

            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                        data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)

                    if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                        # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                        # It can happen with the last batch of the dataset, depending on its size.
                        # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                        text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    else:
                        text_embed, vid_embed = self.model(data, return_embeds=True)

                    text_embed_arr[dl_idx].append(text_embed.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed.cpu())
                    sims_batch = sim_matrix(text_embed, vid_embed)
                    loss = self.loss(sims_batch)
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                        mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
