import os
import json
from datetime import datetime


def file_check(file_name):
    path = os.path.dirname(file_name)
    if not os.path.exists(path):
        os.makedirs(path)

def json_write(data, file_name):
    try:
        file_check(file_name)
        with open(file_name, 'w+') as outfile:
            json.dump(data, outfile)
        print('json file saved at %s'%(file_name))
    except:
        import traceback
        traceback.print_exc()
        print('cannot write %s'%(file_name))
        
def save_vis_re(data, vis_re, save_pth=None, timestamp=True):
    # meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
    # data = {'video': final, 'text': caption, 'meta': meta_arr, 'frame_idxs': idxs}
    # print('save_vis_re:\t', data.keys())
    indices, region_mask = vis_re
    vids = data['meta']['paths']
    raw_caps= data['meta']['raw_captions']
    # TODO support multiple frames
    frame_idxs= data['frame_idxs'][0].tolist()

    re = {}
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
    print(len(vids), indices.size(), region_mask.size())
    for i in range(len(vids)):
        k = str(vids[i])
        v = indices[i].cpu().detach().tolist()
        v1 = region_mask[i].cpu().detach().tolist()
        # print(k,v)
        re[k] = {'cluster_id':v, 'region_mask':v1, 'raw_caption':raw_caps[i], 'frame_idxs':frame_idxs[i]}
        # re[k] = {'cluster_id':v, 'region_mask':v1, 'raw_caption':raw_caps[i]}
        
    # print(re)
    save_pth = os.path.join(save_pth, timestamp)
    try:
        json_write(re, '%s/vis.json'%(save_pth))
    except:
        print("failed to save results!!!")
        import traceback
        traceback.print_exc()