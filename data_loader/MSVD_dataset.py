from base.base_dataset import TextVideoDataset
import pandas as pd
import os


from utils.util import load_json, load_jsonl

class MSVD(TextVideoDataset):
    def _load_metadata(self):
        # metadata_dir = 'meta_data/MSVD'
        split_files = {
            'train': 'MSVD_train.tsv',
            'val': 'MSVD_test.tsv',            # there is no test
            'test': 'MSVD_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))
        # data/MSVD/YouTubeClips

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.avi'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(full_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        return sample[0].split(',')[0]

class MSVD_QA(TextVideoDataset):
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
        ans2label_file = os.path.join(self.metadata_dir, "msvd_answer_set.txt")
        yt_mapping_file = os.path.join(self.metadata_dir, "msvd_youtube_mapping.txt")
        self.ans2label = self.load_ans_set(ans2label_file)
        self.video_mapper = self._get_video_mapper(yt_mapping_file)

        split_files = {
            'train': "msvd_train_qa_encode.json",
            'test': "msvd_test_qa_encode.json",
            'val': "msvd_val_qa_encode.json"
        }# Only top 1000 answers are used

        target_split_fp = split_files[self.split]
        meta_file = os.path.join(self.metadata_dir, target_split_fp)
        metadata = load_json(meta_file)
        data_size = len(metadata)
        if self.subsample < 1:
            data_size = int(data_size * self.subsample)

        self.metadata = metadata[:data_size]
        self.num_labels = len(self.ans2label)
        self.label2ans = {v: k for k, v in self.ans2label.items()}

        print("load split {}, {} samples".format(self.split, data_size))

    def _get_video_path(self, sample):
        video_name = self.video_mapper[str(sample["video_id"])]
        return os.path.join(self.data_dir, video_name + '.avi'), video_name + '.avi'

    def load_ans_set(self, ans_set_file):
        """
        input: A list of answers from a txt file.
        """
        ans2label = {}
        with open(ans_set_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                ans2label[line.strip('\n')] = i

        return ans2label

    def _get_video_mapper(self, mapping_file):
        """
        input: A list of <youtube_id, video_id> from a txt file.
        """
        video_mapper = {}
        with open(mapping_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                yt_id, vid = line.split(' ')
                video_mapper[vid.strip('vid')] = yt_id
        return video_mapper



    def _get_answer_id(self, sample): 
        if sample["answer"] in self.ans2label.keys():
            return self.ans2label[sample["answer"]]
        else:
            return -1 # answers of some test samples may not in vocabulary