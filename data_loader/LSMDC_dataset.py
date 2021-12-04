from base.base_dataset import TextVideoDataset
import pandas as pd
import os


class LSMDC(TextVideoDataset):
    def _load_metadata(self):
        # metadata_dir = 'your_path_to_dataset/LSMDC/meta_data/'
        metadata_dir = './meta_data/LSMDC'


        split_files = {
            # 'train': 'LSMDC16_annos_training.csv',
            'train': 'LSMDC16_annos_training.csv',
            'val': 'LSMDC16_challenge_1000_publictect.csv',
            # 'val': 'LSMDC16_annos_val.csv',            # there is no test
            'test': 'LSMDC16_challenge_1000_publictect.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))
        self.miss_vid_cnt = 0
        # your_path_to_dataset/MSVD/YouTubeClips

    def _get_video_path(self, sample):
        video_fp = sample[0]
        sub_path = video_fp.split('.')[0]
        remove = sub_path.split('_')[-1]
        sub_path = sub_path.replace('_'+remove,'/')
        rel_video_fp = sub_path + video_fp + '.avi'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        
        # if not os.path.exists(full_video_fp):
            # print('full_video_fp:', full_video_fp)
            # self.miss_vid_cnt+=1
            # print('miss_vid_cnt:', self.miss_vid_cnt)
        # assert os.path.exists(full_video_fp)
        
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        return sample[-1] # .split(',')[0]

