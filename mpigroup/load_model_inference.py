from run_videomae_vis_v2 import DataAugmentationForVideoMAEInference, get_model, load_frames, save_video
import yaml
import os.path as osp
import os
from types import SimpleNamespace as Namespace
from typing import List, Union
from timm.models import create_model
import torch
from utils import load_state_dict, time_function_decorator
from torch import sigmoid as logit
# from mpigroup.const import LABELS as LABELS_MAP
# from miga.const import ID2LABELS_SMG_SHORT as LABELS_MAP
import pandas as pd
from PIL import Image
from decord import VideoReader, cpu
import numpy as np

def pars_path(p: Union[str, List[Union[List, str]]]):
    assert isinstance(p, (list, str)), TypeError("p must be a List or a str")

    # If p is a string, return it
    if isinstance(p, str):
        return p

    # If p is an empty list, return an empty string
    if len(p) == 0:
        return ''

    # Initialize an empty list to store components of the path
    components = []

    # Iterate over elements of the nested list
    for item in p:
        # Recursively call pars_path if item is a list
        if isinstance(item, list):
            components.append(pars_path(item))
        # Append the string directly to components if item is a string
        elif isinstance(item, str):
            components.append(item)
        else:
            print(type(item))
            raise TypeError("Invalid type in nested list")

    # Use os.path.join() to construct the path
    return os.path.join(*components)


def get_args(yaml_path, params='finetuning_params'):
    # load yaml
    loaded_config = yaml.safe_load(open(yaml_path, 'r'))
    if params is None:
        params = loaded_config.keys()
    elif isinstance(params, str):
        params = [params]
    all_params = dict()
    for param in params:
        ext_params = loaded_config[param]

        for k, v in ext_params.items():
            if k =='lr':
                pass
            if isinstance(v, list):
                if isinstance(v[0], float):
                    continue
                v = pars_path(v)
            ext_params[k] = v
        all_params.update(ext_params)
    return Namespace(**all_params), all_params




class ModelInference:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.model = self._create_model(args)

        patch_size = self.model.patch_embed.patch_size
        args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
        self.args = args
        self.transform = self._get_data_transforms()


    def _create_model(self, args):
        # load model
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )

        checkpoint = torch.load(args.finetune, map_location='cpu')
        try:
            checkpoint_model = checkpoint['module']
        except:
            checkpoint_model = checkpoint['model']
        load_state_dict(model, checkpoint_model)
        model = model.to(self.device)
        model.eval()
        return model

    def load_video_from_path(self, video_path, frame_id_list=None):
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        # Get 16 frames with 4 sampling rate, this can be more generic by using sampling rate and number of frames
        if frame_id_list is None:
            frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 59]

        return vr.get_batch(frame_id_list).asnumpy()
    def _get_data_transforms(self):
        return DataAugmentationForVideoMAEInference(self.args)

    def _transform_video(self, video_data):
        if isinstance(video_data, list):
            video_data = np.array(video_data)
        n_frames = video_data.shape[0]
        img = [Image.fromarray(video_data[i, :, :, :]).convert('RGB') for i in range(n_frames)]
        # Performe transformations on the image - resizeing, normalization, reshape
        img, _ = self.transform((img, None))  # T*C,H,W
        img = img.view((n_frames, 3) + img.size()[-2:]).transpose(0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        img = img.unsqueeze(0)
        return img
    def run_inference(self, vid, device=None):
        if device is None:
            device = self.device
        vid = self._transform_video(vid)
        vid = vid.to(device)
        out = self.model(vid)
        logits = logit(out).detach().cpu().tolist()
        # df = pd.DataFrame(logits, columns=LABELS_MAP.values()).to_dict()

        return {a:v for a, v in zip(LABELS_MAP.values(), logits[0])}

        # args = get_args(config_path)

if __name__ == '__main__':
    config_path = osp.join('..', 'model_configs', 'mpigroup_multiclass_inference_debug.yaml')
    args, _ = get_args(config_path)
    model = ModelInference(args)

    path_to_video = "D:\\Project-mpg microgesture\\human_micro_gesture_classifier\\video_samples_results\\MPIG_densepose_dual_2\\checkpoint-99\\MPIIGroupInteraction\clips_train\\00000-video\\videos\\ori_vid.mp4"


    vid = model.load_video_from_path(path_to_video, range(16))
    df = model.run_inference(vid)
    print(df)