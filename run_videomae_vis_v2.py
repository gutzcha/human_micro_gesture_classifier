# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
# from datasets import DataAugmentationForVideoMAE    
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
import video_transforms
import volume_transforms
import imageio
import os.path as osp
from modeling_pretrain import PretrainVisionTransformerMultiOutout


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForVideoMAEInference(DataAugmentationForVideoMAE):
    def __init__(self, args):
        super(DataAugmentationForVideoMAEInference, self).__init__(args)
        normalize = GroupNormalize(self.input_mean, self.input_std)

        self.transform = transforms.Compose([
            GroupPadToSquare(),
            GroupScale((args.input_size, args.input_size), interpolation=Image.BILINEAR),
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )


def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', nargs='+', type=str, help='input video path (single or multiple paths)')
    parser.add_argument('--save_path', type=str, help='save video path')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
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

    # ==========================================================
    # ========= Load model =========
    # ========================================================== 

    # When saving:
    # experiment/epoch/dataset name/ dataset type/ file name/ modality (images/videos) / files
    # files-images: ori_image, rec_image, ori_image_dense, rec_image_dense
    # files-videos: ori_video, rec_video, ori_video_dense, rec_video_dense

    # experiment/epoch/ === are given in args.save_path
    # dataset name/ dataset type/ file name === are extracted from the image file path

    # Load model
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

    # Get image transformer - resizeing and normalization
    transformations = DataAugmentationForVideoMAEInference(args)

    # ==========================================================
    # ========= Extract and save video reconstructions =========
    # ========================================================== 

    img_path_list = args.img_path
    if isinstance(img_path_list, str):
        img_path_list = [img_path_list]

    save_folder = args.save_path
    for img_path in img_path_list:

        # Get save folder names and create them
        img_name = osp.join(*(img_path.split('/')[-3:]))[:-4]  # Get image folder folder name

        image_save_path = osp.join(save_folder, img_name, 'images')
        video_save_path = osp.join(save_folder, img_name, 'videos')

        Path(image_save_path).mkdir(parents=True, exist_ok=True)
        Path(video_save_path).mkdir(parents=True, exist_ok=True)

        # Load the video and extract frames
        img, bool_masked_pos, frame_id_list = load_frames(img_path, args.num_frames, transformations)

        if hasattr(args, 'densepose'):
            img_dense_path = img_path.replace('clips', 'densepose')
            img_dense, _, _ = load_frames(img_dense_path, args.num_frames, transformations, frame_id_list=frame_id_list)
        else:
            img_dense = []

        with torch.no_grad():

            # Run inference on model
            outputs = run_inference(model, device, img, bool_masked_pos)
            if isinstance(model, PretrainVisionTransformerMultiOutout):
                # there is more than one output
                features_cfg = model.features_cfg
                model_outputs = {k: {'outputs': v} for k, v in zip(features_cfg.keys(), outputs)}
            else:
                model_outputs = {'videos': {'outputs': outputs}}

            # Recreate original image and get mean and std used for normalization
            ori_img = unnormalize_frames(img, device=device)
            # ==================== save original video ========================
            save_images(ori_img, image_save_path, frame_id_list, prepend='ori_img')
            save_video(ori_img, video_save_path, frame_id_list, prepend='ori_vid')

            # Load and save densepose
            if hasattr(args, 'densepose'):
                ori_img_dense = unnormalize_frames(img_dense, device=device)

                # save_images(densepose_img, image_save_path, frame_id_list, prepend='densepose')
                save_video(ori_img_dense, video_save_path, frame_id_list, prepend='densepose_vid')

            # ============ Reconstruct videos ===================
            for feature, output in model_outputs.items():
                imgs, rec_img, mask = reconstruct_video_from_patches(ori_img, patch_size, bool_masked_pos,
                                                                     output['outputs'], frame_id_list)
                model_outputs[feature]['imgs'] = imgs
                model_outputs[feature]['rec_img'] = rec_img
                model_outputs[feature]['mask'] = mask

            # =========== save reconstruction video ============
            for feature, output in model_outputs.items():
                save_images(output['rec_img'], image_save_path, frame_id_list, prepend=f'{feature}_rec_img')
                save_video(output['rec_img'], video_save_path, frame_id_list, prepend=f'{feature}_rec_vid')

                # if torch.any(bool_masked_pos): # There is a mask
            #     #save masked video if there was masking
            #     img_mask = rec_img * mask
            #     save_images(img_mask, image_save_path, frame_id_list, prepend='mask_img')
            #     save_video(img_mask, video_save_path,frame_id_list, prepend='mask_vid')


def load_frames(img_path, num_frames, transformations, frame_id_list=None):
    with open(img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    # duration = len(vr)
    # new_length  = 1
    # new_step = 1
    # skip_length = new_length * new_step

    # Get 16 frames with 4 sampling rate, this can be more generic by using sampling rate and number of frames
    if frame_id_list is None:
        frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 59]

    video_data = vr.get_batch(frame_id_list).asnumpy()
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

    # Performe transformations on the image - resizeing, normalization, reshape
    img, bool_masked_pos = transformations((img, None))  # T*C,H,W
    img = img.view((num_frames, 3) + img.size()[-2:]).transpose(0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
    bool_masked_pos = torch.from_numpy(bool_masked_pos)
    bool_masked_pos = bool_masked_pos[None, :]
    img = img.unsqueeze(0)
    return img, bool_masked_pos, frame_id_list


def unnormalize_frames(img, device='cpu'):
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
    ori_img = img.to(device) * std + mean  # in [0, 1]
    return ori_img


def run_inference(model, device, img, bool_masked_pos):
    img = img.to(device, non_blocking=True)
    bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
    outputs = model(img, bool_masked_pos)
    return outputs


def save_video(ori_img,
               video_save_path,
               frame_id_list=None,
               prepend='ori_vid',
               vid_ext='mp4',
               fps=7,
               txt=None
               ):
    input_shape = ori_img.shape

    if len(input_shape) == 5:
        # with_batch
        ori_img = ori_img[0]
    if video_save_path.endswith(vid_ext):
        output_path = video_save_path
    else:
        output_path = f"{video_save_path}/{prepend}.{vid_ext}"

    if frame_id_list is None:
        frame_id_list = list(range(ori_img.shape[1]))

    # imgs = [ToPILImage()(ori_img[:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
    imgs = [ToPILImage()(ori_img[:, vid, :, :].cpu().clamp(0, 0.996)) for vid, _ in enumerate(frame_id_list)]
    if txt:
        font = ImageFont.load_default()  # You can choose any font you prefer
        # text_color = (255, 255, 255)  # White color for the text
        text_color = (255, 0, 0)  # White color for the text

        # debug
        # print(imgs[0].height)
        txt_h = 20
        txt_w = 20
        h = 10
        for i, img in enumerate(imgs):
            draw = ImageDraw.Draw(img)
            # debug
            # print(txt)
            # hh = txt_h+i*h
            # print(hh)
            draw.text((txt_w, txt_h), txt, fill=text_color, font=font, align="left")

    image_array_list = [np.array(img) for img in imgs]

    # Write the images to an MP4 file
    writer = imageio.get_writer(output_path, fps=fps)
    for image_array in image_array_list:
        writer.append_data(image_array)
    writer.close()


def save_images(ori_img, image_save_path, frame_id_list, prepend='ori_img'):
    if frame_id_list is None:
        frame_id_list = list(range(ori_img.shape[1]))
    input_shape = ori_img.shape

    if len(input_shape) == 5:
        # with_batch
        ori_img = ori_img[0]

    # imgs = [ToPILImage()(ori_img[:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)]
    imgs = [ToPILImage()(ori_img[:, vid, :, :].cpu().clamp(0, 0.996)) for vid, _ in enumerate(frame_id_list)]
    for id, im in zip(frame_id_list, imgs):
        im.save(f"{image_save_path}/{prepend}.{id:03d}.jpg")


def reconstruct_video_from_patches(
        ori_img, patch_size, bool_masked_pos, outputs, frame_id_list, normalize_with_orig=True):
    img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0],
                            p2=patch_size[0])
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)').clone()

    if not (bool_masked_pos is None) and torch.any(bool_masked_pos):  # There is a mask
        # make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0],
                         p2=patch_size[1], h=14, w=14)
        B, T, C, W, H = ori_img.shape
        img_patch[bool_masked_pos] = outputs.reshape(-1, outputs.shape[-1])
        # img_patch = img_patch.reshape(B,-1,outputs.shape[-1])
        img_patch[bool_masked_pos] = 0
    else:
        img_patch = outputs
        mask = []

    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    if frame_id_list is None:
        frame_id_list = range(ori_img.shape[2])

    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    if normalize_with_orig:
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(
            dim=-2, keepdim=True)
    rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0],
                        p2=patch_size[1], h=14, w=14)
    imgs = [ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0, 0.996)) for vid, _ in enumerate(frame_id_list)]

    return imgs, rec_img, mask


def save_list_of_images_as_video(image_array_list, output_path, fps):
    # Write the images to an MP4 file
    writer = imageio.get_writer(output_path, fps=fps)
    for image_array in image_array_list:
        writer.append_data(image_array)
    writer.close()


if __name__ == '__main__':
    opts = get_args()
    main(opts)
