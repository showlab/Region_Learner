from base.base_dataset import TextVideoDataset

import pandas as pd
import os
import json
import numpy as np
import random


class UCF(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes UCF101 data is structured as follows.
    UCF101/
        videos/
                1.avi           (videoid.mp4)
                ...
                5000.avi
            ...
    """
    def _load_metadata(self):
        metadata_dir = os.path.join(self.metadata_dir, 'metadata')
        metadata_fp = os.path.join(metadata_dir, f'{self.split}_{self.cut}.csv')
        metadata = pd.read_csv(metadata_fp)


        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.


        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        self.metadata['action_str'] = self.metadata['action_str'].str[:350]

    def _get_video_path(self, sample):
        # rel_video_fp = os.path.join(sample['action_str'], str(sample['video_name']))
        rel_video_fp = str(sample['video_name'])
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        action = sample['action_str'].replace('_',' ')
        if self.split=='train':
            cap_list = [
                f"A video of {action}",
                f"A person is {action}"
            ]
            return random.choice(cap_list)
        else:
            # return f"A person is {action}"
            return action

