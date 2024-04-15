# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
# from datasets import DataAugmentationForVideoMAE    
from run_videomae_vis import DataAugmentationForVideoMAE, DataAugmentationForVideoMAEInference
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import  TubeMaskingGenerator
import video_transforms
import volume_transforms
import imageio
import glob
import os.path as osp
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', type=str, help='input video folder path')
    parser.add_argument('--save_path', type=str, help='save video path')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model

def load_encoder(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.encoder
    model.eval()
    return model, args

def load_classifier():
    pass



    pass
def main(args):
    print(args)

    


    video_folder = args.img_path
    list_of_videos = glob.glob(osp.join(video_folder,'*.mp4'))
    for img_path in tqdm(list_of_videos):

       # Replace "_SPLIT" with "_FEATURES/VideoMAE_ViT-B_checkpoint_Kinetics-400" in the video path

        new_video_path = img_path.replace("_SPLIT", "_FEATURES/VideoMAE_ViT-B_checkpoint_Kinetics-400")

        # Extract file name and new folder name
        file_name = Path(new_video_path).stem
        save_path = Path(new_video_path).parent / 'features'
        

        Path(save_path).mkdir(parents=True, exist_ok=True)
        

        with open(img_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        duration = len(vr)
        new_length  = 1 
        new_step = 1
        skip_length = new_length * new_step
        frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 59]
        if duration > frame_id_list[-1]:
            frame_id_list = [c if c < duration else duration-1  for c in frame_id_list]



        video_data = vr.get_batch(frame_id_list).asnumpy()
        # print(video_data.shape)
        img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

        transforms = DataAugmentationForVideoMAEInference(args)
        # transforms = DataAugmentationForVideoMAE(args)
        img, bool_masked_pos = transforms((img, None)) # T*C,H,W
        # print(img.shape)
        img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        # img = img.view(( -1 , args.num_frames) + img.size()[-2:]) 
        bool_masked_pos = torch.from_numpy(bool_masked_pos)

        with torch.no_grad():
            # img = img[None, :]
            # bool_masked_pos = bool_masked_pos[None, :]
            img = img.unsqueeze(0)
            # print(img.shape)
            bool_masked_pos = bool_masked_pos.unsqueeze(0)
            
            img = img.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
            
            features = model.encoder(img, bool_masked_pos).view(-1).cpu().numpy()
           

            output_path = Path(save_path) / f"feat_{file_name}.npy"
            np.save(output_path, features)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
