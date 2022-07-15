import argparse
import collections
import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from utils.util import replace_nested_dict_item
from parse_config import ConfigParser
from trainer import Multi_Trainer_dist
from sacred import Experiment
# from neptunecontrib.monitoring.sacred import NeptuneObserver
import transformers
import os

ex = Experiment('train')

@ex.main
def run():
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ['TRANSFORMERS_OFFLINE'] = "1"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_address, args.master_port),
                                         rank=args.rank, world_size=args.world_size)
    device = torch.device(f'cuda:{args.local_rank}')
    if args.rank == 0:
        print('world_size', args.world_size, flush=True)
        print('local_rank: ', args.local_rank, flush=True)
    # build tokenizer
    text_params = config['arch']['args']['text_params']
    # print('here', '/'.join((text_params['pretrained_path'], text_params['model'])))
    if 'CLIP' in text_params['model']:
        tokenizer = None
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained('/'.join((text_params['pretrained_path'], text_params['model'])),
                                                           TOKENIZERS_PARALLELISM=False)
    
    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    # dataset_name = config['data_loader'][0]['args']['dataset_name']
    # if dataset_name in ['MSVD', 'DiDeMo', 'LSMDC'] and False:
    #     data_loader, valid_data_loader = init_dataloaders(config, module_data, mode='val')
    #     # get test data loader by the same function
    #     _, test_data_loader = init_dataloaders(config, module_data, mode='test')
    # else:
    #     data_loader, valid_data_loader = init_dataloaders(config, module_data, mode='val')


    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
        # if dataset_name in ['MSVD', 'DiDeMo', 'LSMDC'] and False:
            # print('Test dataset: ', [x.n_samples for x in test_data_loader], ' samples')

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    if args.local_rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # for name, param in model.named_parameters(): #查看可优化的参数有哪些
    #     if param.requires_grad:
    #         print(name)
    # exit(0)
    # print('train', args)
    if 'clip' in args.config:
        trainable_params = [
            {"params": model.clip_model.parameters(), "lr": 1e-7},
            # {"params": model.text_model.parameters()},
            # {"params": model.txt_proj.parameters()},
            # {"params": model.vid_proj.parameters()},
        ]
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None
    trainer = Multi_Trainer_dist(args, model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])
    trainer.train()


def init_dataloaders(config, module_data, mode='val'):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', mode)
        new_data_loader = [config.initialize("data_loader", module_data)]
        
    elif isinstance(config["data_loader"], list):
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', mode)
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        new_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]

    else:
        raise ValueError("Check data_loader config, not correct format.")

    # return data_loader, valid_data_loader
    return data_loader, new_data_loader
    


if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')     
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    args.add_argument('-k', '--local_rank', type=int, default=0)

    master_address = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    world_size = int(os.environ['WORLD_SIZE'])
    # world_size = int(torch.cuda.device_count())
    rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    #### modified by Mr Yan
    # if args.rank == 0:
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")

    ###
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    # args.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    # args.add_argument('-sc', '--schedule', default=[60, 80])
    args.add_argument('-sc', '--schedule', default=[60, 80], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
    args.add_argument('-de', '--debug', dest='debug', action='store_true')
    args.add_argument('-vs', '--vis_saving', dest='vis_saving', action='store_true')
    args.add_argument('-ar', '--auto_resume', dest='auto_resume', action='store_true')





    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        
        #######################################################################################################
        ## added by Mr. Yan
        # if you do not pass values to them, these args will be read from config files
        CustomArgs(['--model'], type=str, target=('arch', 'args', 'video_params', 'model')),
        CustomArgs(['--random_sampling'], type=bool, target=('arch', 'args', 'video_params', 'random_sampling')),
        CustomArgs(['--VQ_num_tokens'], type=int, target=('arch', 'args', 'video_params', 'VQ_num_tokens')),
        CustomArgs(['--AGG_region_num'], type=int, target=('arch', 'args', 'video_params', 'AGG_region_num')),
        CustomArgs(['--Interaction_depth'], type=int, target=('arch', 'args', 'video_params', 'Interaction_depth')),
        CustomArgs(['--temporal_type'], type=str, target=('arch', 'args', 'video_params', 'temporal_type')),
        
        # debug now
        CustomArgs(['--Motion_Excitation'], type=bool, target=('arch', 'args', 'video_params', 'Motion_Excitation')),
        # CustomArgs(['--RL_before_pos'], type=int, target=('arch', 'args', 'video_params', 'RL_before_pos')),
        



        # for the first dataset
        CustomArgs(['--bs_0', '--batch_size_0'], type=int, target=('data_loader', 0, 'args', 'batch_size')),
        CustomArgs(['--num_workers_0'], type=int, target=('data_loader', 0, 'args', 'num_workers')),
        CustomArgs(['--data_dir_0'], type=str, target=('data_loader', 0, 'args', 'data_dir')),
        CustomArgs(['--num_frames_0'], type=int, target=('data_loader', 0, 'args', 'video_params', 'num_frames')),

        # for the second dataset
        CustomArgs(['--bs_1', '--batch_size_1'], type=int, target=('data_loader', 1, 'args', 'batch_size')),
        CustomArgs(['--num_workers_1'], type=int, target=('data_loader', 1, 'args', 'num_workers')),
        CustomArgs(['--data_dir_1'], type=str, target=('data_loader', 1, 'args', 'data_dir')),
        CustomArgs(['--num_frames_1'], type=int, target=('data_loader', 1, 'args', 'video_params', 'num_frames')),

        CustomArgs(['--use_amp'], type=bool, target=('trainer', 'use_amp')),
        CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--accum_iter'], type=int, target=('trainer', 'accum_iter')),

        CustomArgs(['--save_dir'], type=str, target=('trainer', 'save_dir')),
        CustomArgs(['--load_checkpoint'], type=str, target=('arch', 'args', 'load_checkpoint')),

        #######################################################################################################
    ]




    config = ConfigParser(args, options)
    args = args.parse_args()
    ex.add_config(config._config)

    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError('Neptune credentials not set up yet.')
        ex.observers.append(NeptuneObserver(
            api_token='INSERT TOKEN',
            project_name='INSERT PROJECT NAME'))
        ex.run()
    else:
        run()
