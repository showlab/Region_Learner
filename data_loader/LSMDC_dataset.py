from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import torch

class LSMDC(TextVideoDataset):
    def _load_metadata(self):
        # metadata_dir = './meta_data/LSMDC'
        split_files = {
            # 'train': 'LSMDC16_annos_training.csv',
            'train': 'LSMDC16_annos_training.csv',
            'val': 'LSMDC16_challenge_1000_publictect.csv',
            # 'val': 'LSMDC16_annos_val.csv',            # there is no test
            'test': 'LSMDC16_challenge_1000_publictect.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))
        self.miss_vid_cnt = 0

    def _get_video_path(self, sample):
        video_fp = sample[0]
        sub_path = video_fp.split('.')[0]
        remove = sub_path.split('_')[-1]
        sub_path = sub_path.replace('_'+remove,'/')
        rel_video_fp = sub_path + video_fp + '.avi'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        return sample[-1] # .split(',')[0]



class LSMDC_MC(TextVideoDataset):
    def __getitem__(self, idx):
        ######################## Just copy from base #############################
        idx = idx % len(self.metadata)
        sample = self.metadata[idx] # train/test for list
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
        # print(self.split, caption)

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        # data = {'video': final, 'text': caption, 'meta': meta_arr, 'frame_idxs': idxs}
        ######################## Just copy from base #############################


        data = {'video': self.get_video(sample, video_fp), 'text': caption, 'meta': meta_arr, 'answer': answer}
        # print('base_dataset:\t', data.keys())
        return data

    def _load_metadata(self):
        # metadata_dir =  os.path.join(self.metadata_dir, "meta_data")
        metadata_dir = self.metadata_dir
        split_files = {
            'train': 'LSMDC16_multiple_choice_train.csv',
            'val': 'LSMDC16_multiple_choice_test_randomized.csv',            # there is no test
            'test': 'LSMDC16_multiple_choice_test_randomized.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)

        datalist = []
        for raw_id in range(len(metadata)):
            raw_d = metadata.iloc[raw_id]
            video_fp = raw_d[0]
            options = [raw_d[idx] for idx in range(5, 10)]
            d = dict(
                id=video_fp,
                # vid_id=rel_video_fp,
                answer=raw_d[-1] - 1 if self.split in ['val', 'test'] else 0,
                options=options,
            )
            datalist.append(d)
        self.metadata = datalist
        # self.id2answer = {d["id"]: int(d["answer"]) for d in self.metadata}
        # self.id2data = {d["id"]: d for d in self.metadata}
        print("load split {}, {} samples".format(self.split, len(metadata)))


    def _get_video_path(self, sample):
        video_fp = sample['id']
        sub_path = video_fp.split('.')[0]
        remove = sub_path.split('_')[-1]
        sub_path = sub_path.replace('_'+remove,'/')
        rel_video_fp = sub_path + video_fp + '.avi'
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        # print('_get_video_path', full_video_fp, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample['options'][0] # for train only, select the first option as cap for RET. 