from base.base_dataset import TextVideoDataset
import pandas as pd
import os
import json
import numpy as np
import random


# class WebVid(TextVideoDataset):
#     """
#     WebVid Dataset.
#     Assumes webvid data is structured as follows.
#     Webvid/
#         videos/
#             000001_000050/      ($page_dir)
#                 1.mp4           (videoid.mp4)
#                 ...
#                 5000.mp4
#             ...
#     """
#     def _load_metadata(self):
#         # metadata_dir = './meta_data/WEBVID/'

#         # NOTE: Not all videos can be download. Maybe you need adjust the meta_data for each dataset
#         split_files = {            
#             'train': 'results_2M_train.csv',
#             'val': 'results_2M_val.csv',            # there is no test
#         }
#         target_split_fp = split_files[self.split]
#         metadata = pd.read_csv(os.path.join(self.metadata_dir, target_split_fp), sep='\t')
#         if self.subsample < 1:
#             metadata = metadata.sample(frac=self.subsample)

#         # modified by Mr. Yan. Use full val set.
#         # elif self.split == 'val':
#         # metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

#         #metadata['caption'] = metadata['name']
#         #del metadata['name']
#         self.metadata = metadata
#         # TODO: clean final csv so this isn't necessary
#         #self.metadata.dropna(inplace=True)
#         #self.metadata['caption'] = self.metadata['caption'].str[:350]

#     def _get_video_path(self, sample):

#         # rel_video_fp = sample[1] + '.mp4'
#         rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
#         full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)

#         return full_video_fp, rel_video_fp

#     def _get_caption(self, sample):
#         return sample[0]


class WebVid(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def _load_metadata(self):
        metadata_dir = os.path.join(self.metadata_dir, 'metadata')
        metadata_fp = os.path.join(metadata_dir, f'results_{self.cut}_{self.split}.csv')
        metadata = pd.read_csv(metadata_fp)

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)

        # need all val samples
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample['caption']