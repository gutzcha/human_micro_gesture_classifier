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


def main(args):
    print(args)

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
    model.eval()
    video_folder = args.img_path
    list_of_videos = glob.glob(osp.join(video_folder,'*.mp4'))
    for img_path in tqdm(list_of_videos):

       # Replace "_SPLIT" with "_RECONSTRUCTED" in the video path
        new_video_path = img_path.replace("_SPLIT", "_RECONSTRUCTED")

        # Extract file name and new folder name
        file_name = Path(new_video_path).stem
        save_path = Path(new_video_path).parent / 'images'
        video_path = Path(new_video_path).parent / 'videos'

        Path(save_path).mkdir(parents=True, exist_ok=True)
        Path(video_path).mkdir(parents=True, exist_ok=True)

        with open(img_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        duration = len(vr)
        new_length  = 1 
        new_step = 1
        skip_length = new_length * new_step
        frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 59]
        if duration > frame_id_list[-1]:
            frame_id_list = [c if c < duration else duration-1  for c in frame_id_list]


        
        # tmp = np.arange(0,32, 2) + 60
        # frame_id_list = tmp.tolist()
        # average_duration = (duration - skip_length + 1) // args.num_frames
        # if average_duration > 0:
        #     frame_id_list = np.multiply(list(range(args.num_frames)),
        #                             average_duration)
        #     frame_id_list = frame_id_list + np.random.randint(average_duration,
        #                                             size=args.num_frames)

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
            outputs = model(img, bool_masked_pos)

            #save original video
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            ori_img = img * std + mean  # in [0, 1]
            imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
            for id, im in enumerate(imgs):
                save_image_path = Path(save_path) / f"ori_{file_name}_{id}.jpg"
                im.save(save_image_path)
            
            image_array_list = [np.array(img) for img in imgs]
            output_path = Path(video_path) / f"ori_{file_name}.mp4"
            # Write the images to an MP4 file
            writer = imageio.get_writer(output_path, fps=7)
            for image_array in image_array_list:
                writer.append_data(image_array)
            writer.close()

        
            img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
            img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
            
            
            if torch.any(bool_masked_pos): # There is a mask
                #make mask
                mask = torch.ones_like(img_patch)
                mask[bool_masked_pos] = 0
                mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
                mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
                img_patch[bool_masked_pos] = outputs
            else:
                img_patch = outputs

            #save reconstruction video
            rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
            # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
            rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
            rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
            imgs = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]

            for id, im in enumerate(imgs):
                save_image_path = Path(save_path) / f"rec_{file_name}_{id}.jpg"
                im.save(save_image_path)
            
            image_array_list = [np.array(img) for img in imgs]
            output_path = Path(video_path) / f"rec_{file_name}.mp4"

            # Write the images to an MP4 file
            writer = imageio.get_writer(output_path, fps=7)
            for image_array in image_array_list:
                writer.append_data(image_array)
            writer.close()

            if torch.any(bool_masked_pos): # There is a mask

                #save masked video 
                img_mask = rec_img * mask
                imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
                for id, im in enumerate(imgs):
                    save_image_path = Path(save_path) / f"mask_{file_name}_{id}.jpg"
                    im.save(save_image_path)                    
                
                image_array_list = [np.array(img) for img in imgs]
                output_path = Path(video_path) / f"mask_{file_name}.mp4"
                # Write the images to an MP4 file
                writer = imageio.get_writer(output_path, fps=7)
                for image_array in image_array_list:
                    writer.append_data(image_array)
                writer.close()

if __name__ == '__main__':
    opts = get_args()
    main(opts)
