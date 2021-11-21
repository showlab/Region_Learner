from base.base_dataset import TextVideoDataset
import pandas as pd
import os


class MSVD(TextVideoDataset):
    def _load_metadata(self):
        metadata_dir = 'meta_data/MSVD'
        split_files = {
            'train': 'MSVD_train.tsv',
            'val': 'MSVD_test.tsv',            # there is no test
            'test': 'MSVD_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))
        # /cfs/cfs-4260a4096/260a4096/public_datasets/MSVD/YouTubeClips

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.avi'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(full_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        return sample[0].split(',')[0]

