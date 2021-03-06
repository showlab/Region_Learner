import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, \
    BertTokenizer, T5EncoderModel
import argparse
import torch
import timm
import os
from model.video_transformer import SpaceTimeTransformer
import torchvision

# added by Mr. Yan
from model.helper import get_random_sample_indices
from model.helper import save_vis_re
from model.qa_model import BUTDQAHead
from model.clip import clip
device = "cuda" if torch.cuda.is_available() else "cpu"

class FrozenInTime(BaseModel):
    def __init__(self,
                 args,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()
        self.args = args
        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        print('here', '/'.join((text_params['pretrained_path'], text_params['model'])))
        if text_params['model'] == "CLIP":
            # self.clip_tokenizer = ClipTokenizer()
            pass # we load the whole clip following 
        else:
            self.text_model = AutoModel.from_pretrained('/'.join((text_params['pretrained_path'], text_params['model'])))
            self.text_model.train()


        # TODO: need a better arg as the condition
        num_ans = video_params.get('num_ans', -1)
        if num_ans>0:
            self.QA_head = BUTDQAHead(v_dim=projection_dim, q_dim=projection_dim, hid_dim=projection_dim, num_ans=num_ans)
        else:
            self.QA_head = None

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            if arch_config == 'base_patch16_224':
                vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                model = SpaceTimeTransformer(num_frames=num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style,
                                            video_params=video_params) # try by Mr. Yan
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                # vit_checkpoint = vit_model.state_dict()
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model

            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()

        elif video_params['model'] == "CLIP":
            model, preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model = model.float().train()
            ftr_dim = model.visual.output_dim
            # image_features = model.encode_image(image)
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")



        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(self.args.local_rank))
            #checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            # self.load_state_dict(new_state_dict, strict=True)
            # print(vid_proj)
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            print('missing_keys:\t', missing_keys)
            print('unexpected_keys:\t', unexpected_keys)

            

    def set_device(self, device):
        self.device = device

    def forward_test(self, video):
        video_embeddings = self.compute_video(video)
        return video_embeddings

    def forward(self, data, return_embeds=True, epoch=0):
        # meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        # data = {'video': final, 'text': caption, 'meta': meta_arr, 'frame_idxs': idxs}
        text_data = data['text']
        video_data = data['video']
        # meta_data = data['meta']

        text_embeddings = self.compute_text(text_data)
        # with torch.autograd.set_detect_anomaly(True):
        video_embeddings, tmp_re = self.compute_video(video_data, epoch)

        # added by Mr. YAN
        if self.args.vis_saving and self.args.debug:
            save_vis_re(data, tmp_re, save_pth='tmp_result/vis')
            exit(0)

        # for QA
        if self.QA_head:
            # QA head is a simple fc
            return self.QA_head(video_embeddings, text_embeddings)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif 'CLIP' in self.text_params['model']:
            text_data = clip.tokenize(text_data, truncate=True).to(device)
            text_embeddings = self.clip_model.encode_text(text_data)
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings






    def compute_video(self, video_data, epoch=0):
        T = video_data.size(1)
        ##################### reshape inputs ####################
        # added by Mr. Yan
        if 'resnet' in self.video_params['model']:
            # no temporal
            # print('The size of input video data:', video_data.size())
            video_data = video_data.view(-1, *video_data.size()[2:])
        elif 'MViT' in self.video_params['model']:
            # print('The size of input video data:', video_data.size()) # B, T, C, H, W
            video_data = video_data.transpose(1, 2) # B, C, T, H, W
        elif 'CLIP' in self.video_params['model']:
            # print('The size of input video data:', video_data.size()) 
            video_data = video_data.view(-1, *video_data.size()[2:]) # B*T, C, H, W
        
        ########################## forward #######################
        if 'CLIP' in self.video_params['model']:
            # print('The size of input video data:', video_data.size()) 
            # print('The res of model:', self.clip_model.visual.input_resolution) 
            video_embeddings = self.clip_model.encode_image(video_data)
            if T>1:
                # print('The size of video_embeddings 1:', video_embeddings.size())
                video_embeddings = video_embeddings.reshape(-1, T, *video_embeddings.size()[1:])
                # print('The size of video_embeddings 2:', video_embeddings.size())
                video_embeddings = torch.mean(video_embeddings, 1).squeeze()
                # print('The size of video_embeddings 3:', video_embeddings.size())
            tmp_re = None
        else:
            if 'Transformer' in self.video_params['model']:
                video_embeddings, tmp_re = self.video_model(video_data, epoch)
            else:
                video_embeddings, tmp_re = self.video_model(video_data)

        # print('video_embeddings size:\t', video_embeddings.size())

        
        ################## reshape outputs ######################
        if 'resnet' in self.video_params['model']:
            video_embeddings = video_embeddings.view(*video_embeddings.size()[:2], -1)
            # do random sampling here for data aug??
            # print('random_sampling, training:', self.video_params['random_sampling'], self.training)
            # if self.video_params['random_sampling'] and self.training:
            if self.video_params['random_sampling'] and self.training:
                # print('video_embeddings size:\t', video_embeddings.size())
                sampled_indices = get_random_sample_indices(
                    seq_len=video_embeddings.shape[-1],
                    device=video_embeddings.device)

                # video_embeddings[:,:,:,sampled_indices] = 0
                video_embeddings = video_embeddings.index_select(
                    dim=-1, index=sampled_indices)  # (B, #samples, d)
                # print('video_embeddings size:\t', video_embeddings.size())
            video_embeddings = self.new_pool(video_embeddings)
            video_embeddings = video_embeddings.squeeze()
        # exit(0)
        # print('video_embeddings size:\t', video_embeddings.size())
        elif 'MViT' in self.video_params['model']:
            pass
    
        video_embeddings = self.vid_proj(video_embeddings)


        return video_embeddings, tmp_re

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt



# added by Mr. Yan
# def load_part_state_dict(model, pretrained_dict):
#     model_dict = model.state_dict()
#     ## print(model_dict)
#     # 1. filter out unnecessary keys
#     pretrained_dict = {}
#     for k, v in pretrained_dict.items():
#         if k in model_dict:
#             pretrained_dict[k] = v
#         else:
#             print(k, 'can not exsiting current model!')
#     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)


if __name__ == "__main__":
    pass
