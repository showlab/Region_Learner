from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import json
import numpy as np
import random

import json
import torch

from utils.util import load_json, load_jsonl


class MSRVTT(TextVideoDataset):
    def _load_metadata(self):
        json_fp = os.path.join(self.metadata_dir, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(self.metadata_dir, 'high-quality', 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        if self.cut == "miech":
            train_list_path = "train_list_miech.txt"
            test_list_path = "test_list_miech.txt"
        elif self.cut == "jsfusion":
            train_list_path = "train_list_jsfusion.txt"
            test_list_path = "val_list_jsfusion.txt"
            js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
        elif self.cut in {"full-val", "full-test"}:
            train_list_path = "train_list_full.txt"
            if self.cut == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        elif self.cut in challenge_splits:
            train_list_path = "train_list.txt"
            if self.cut == "val":
                test_list_path = f"{self.cut}_list.txt"
            else:
                test_list_path = f"{self.cut}.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(self.cut))

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']

        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption


class MSRVTT_MC(TextVideoDataset):
    def __getitem__(self, idx):
        ######################## Just copy from base #############################
        idx = idx % len(self.metadata)
        sample = self.metadata.iloc[idx] if self.split=='train' else self.metadata[idx] # test for jsonl
        video_fp, rel_fp = self._get_video_path(sample)
        ######################## Just copy from base #############################

        if self.split == 'train':
            # just do RET
            caption = self._get_caption(sample)
            answer = -1
        else:
            # for MC choice Test, caption is the concated text of multiple choices
            # caption = {}
            # for i, opt in enumerate(sample["options"]):
            #     caption[i] = opt
            caption = sample["options"]
            # print('caption:', len(caption), caption)
            answer = int(sample["answer"])


        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': self.get_video(sample, video_fp), 'text': caption, 'meta': meta_arr, 'answer': answer}
        # print('base_dataset:\t', data.keys())
        return data

    # added by Mr. YAN
    def _load_metadata(self):
        if self.split=='train':
            print('here is train', self.data_dir)
            self.metadata = self._load_ret_train_full_metadata(self.data_dir)
        elif self.split=='val':
            meta_file = os.path.join(self.metadata_dir, "mc_test.jsonl")
            metadata = load_jsonl(meta_file)
            data_size = len(metadata)
            if self.subsample < 1:
                data_size = int(data_size * self.subsample)
            self.metadata = metadata[:data_size]

        print("load split {}, {} samples".format(self.split, len(self.metadata)))

    def _load_ret_train_full_metadata(self, metadata_dir):
        # copy from training part of ret metadata loading
        json_fp = os.path.join(metadata_dir, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(metadata_dir, 'high-quality', 'structured-symlinks')
        train_list_path = "train_list_full.txt"
        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        df = df[df['image_id'].isin(train_df['videoid'])]
        metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)

        return pd.DataFrame({'captions': metadata})

    def _get_video_path(self, sample):
        if self.split=='train':
            return os.path.join(self.data_dir, 'videos', 'all', sample.name + '.mp4'), sample.name + '.mp4'
        else:
            return os.path.join(self.data_dir, 'videos', 'all', sample["clip_name"] + '.mp4'), sample["clip_name"] + '.mp4'
            


    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption

class MSRVTT_QA(TextVideoDataset):
    def __getitem__(self, idx):
        idx = idx % len(self.metadata)
        sample = self.metadata[idx]
        video_fp, rel_fp = self._get_video_path(sample)
        question = sample['question']
        answer_id = self._get_answer_id(sample)
        
        meta_arr = {'raw_captions': question, 'paths': rel_fp, 'dataset': self.dataset_name}
        # data = {'video': final, 'text': caption, 'meta': meta_arr, 'frame_idxs': idxs}
        data = {'video': self.get_video(sample, video_fp), 'text': question, 'meta': meta_arr, 'answer_id': answer_id}
        # print('base_dataset:\t', data.keys())
        return data

    # added by Mr. YAN
    def _load_metadata(self):
        ans2label_file = os.path.join(self.metadata_dir, "msrvtt_train_ans2label.json")
        self.ans2label = load_json(ans2label_file)
        split_files = {
            'train': "msrvtt_qa_train.jsonl",
            'test': "msrvtt_qa_test.jsonl",
            'val': "msrvtt_qa_val.jsonl"
        }
        target_split_fp = split_files[self.split]
        meta_file = os.path.join(self.metadata_dir, target_split_fp)
        metadata = load_jsonl(meta_file)
        data_size = len(metadata)
        if self.subsample < 1:
            data_size = int(data_size * self.subsample)

        self.metadata = metadata[:data_size]
        self.num_labels = len(self.ans2label)
        self.label2ans = {v: k for k, v in self.ans2label.items()}

        print("load split {}, {} samples".format(self.split, data_size))

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample["video_id"] + '.mp4'), sample["video_id"] + '.mp4'

    def _get_answer_id(self, sample): 
        if sample["answer"] in self.ans2label.keys():
            return self.ans2label[sample["answer"]]
        else:
            return -1 # answers of some test samples may not in vocabulary