from base import BaseDataLoader, BaseDataLoaderExplicitSplit, DistBaseDataLoaderExplicitSplit, MultiDistBaseDataLoaderExplicitSplit, BaseMultiDataLoader
from data_loader.transforms import init_transform_dict, init_video_transform_dict
from data_loader.ConceptualCaptions_dataset import ConceptualCaptions3M
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.MSVD_dataset import MSVD
from data_loader.DiDeMo_dataset import DiDeMo
from data_loader.LSMDC_dataset import LSMDC
from data_loader.WebVid_dataset import WebVid


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='cv2'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    elif dataset_name == "DIDEMO":
        dataset = DiDeMo(**kwargs)
    elif dataset_name == "SomethingSomethingV2":
        dataset = SomethingSomethingV2(**kwargs)
    elif dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "ConceptualCaptions12M":
        dataset = ConceptualCaptions12M(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    elif dataset_name == "COCOCaptions":
        dataset = COCOCaptions(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False

        


        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class DistTextVideoDataLoader(DistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        
        # BUG repaired by Mr. YAN
        if video_params['num_frames'] > 1:
            # video data can not do flip, crop aug
            tsfm_dict = init_video_transform_dict(**tsfm_params)
        else:
            tsfm_dict = init_transform_dict(**tsfm_params)

        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class MultiDistTextVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        # tsfm_dict = init_transform_dict(**tsfm_params)

        # BUG repaired by Mr. YAN
        if video_params['num_frames'] > 1:
            # video data can not do flip, crop aug
            tsfm_dict = init_video_transform_dict(**tsfm_params)
        else:
            tsfm_dict = init_transform_dict(**tsfm_params)

        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)
