import os.path as osp
import os

import pandas as pd
from torchvision import transforms
from torchvision.io import read_video
# from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

from einops import rearrange
from modeling_pretrain import (
    pretrain_videomae_small_patch16_224,
    pretrain_videomae_base_patch16_224,
    pretrain_videomae_huge_patch16_224
)

import glob
from types import SimpleNamespace
from datasets import build_dataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms

IMG_STD = [0.229, 0.224, 0.225]
IMG_MEAN = [0.485, 0.456, 0.406]


def pad_frames(vid, out_len, method='reflect'):
    len_vid = len(vid)
    if out_len < len_vid:
        return vid
    ret = []
    if method == 'reflect':
        pad_size = out_len - len_vid
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        for il in range(pad_right):
            ret.append(vid[0])
        ret += vid
        for ir in range(pad_right):
            ret.append(vid[-1])
    return ret


num_frames = 16


def load_video(video_path, num_frames=16):
    # Load the video
    # video, audio, info = read_video(video_path, output_format="TCHW")
    video, audio, info = read_video(video_path, output_format="THWC")
    video_shape = video.shape

    # Apply the transformation to each frame in the video
    # frames_resized = [transform(frame) for frame in video]
    # frames_permuted = [frame.permute(1, 2, 0) for frame in frames_resized]
    # frames_permuted = [frame.permute(1, 2, 0) for frame in video]
    frames_permuted = video
    frames_permuted = pad_frames(frames_permuted, num_frames, method='reflect')

    return frames_permuted


# def load_model_hf(model_name='MCG-NJU/videomae-base-short-ssv2'):
#     feature_extractor = VideoMAEImageProcessor.from_pretrained(model_name)
#     model = VideoMAEForPreTraining.from_pretrained(model_name)
#     return model, feature_extractor


# def run_model_hf(pixel_values, model, p=0.5):
#     bool_masked_pos = get_mask(model, p)
#     outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#     return outputs, pixel_values


def get_mask(p=0.5, method='random', config=None):
    if p == 0:
        return None

    if config is None:
        img_size = 224
        patch_size = 16
    else:
        img_size = config.input_size
        patch_size = config.patch_size[0]

    tubelet_size = 2
    num_patches_per_frame = (img_size // patch_size) ** 2
    seq_length = (num_frames // tubelet_size) * num_patches_per_frame
    if method == 'random':
        return torch.rand(1, seq_length) < p
    else:
        num_true = int(p * 14)
        mask = torch.zeros(1, seq_length, dtype=torch.bool)
        mask = rearrange(mask, 'b (t h w) -> b t h w', t=8, h=14, w=14)
        if method == 'left':
            mask[:, :, :, :num_true] = True
        elif method == 'right':
            mask[:, :, :, -num_true:] = True
        elif method == "top":
            mask[:, :, :num_true, :] = True
        elif method == 'bottom':
            mask[:, :, -num_true:, :] = True
        elif method == 'horizontal':
            mask[:, :, ::2, :] = True
        elif method == 'vertical':
            mask[:, :, :, ::2] = True
        elif method == 'last':
            mask[:, -1, :, :] = True
        elif method == 'mid':
            mask[:, 3, :, :] = True
        else:
            raise f'unknown method {method}'

        mask = rearrange(mask, 'b t h w -> b (t h w)', t=8, h=14, w=14)

    return mask


def run_model_mine(pixel_values, model, config, p=0.9, method='random', reconstruction_mode=False):
    if p == 0:
        bool_masked_pos = None
    else:
        bool_masked_pos = get_mask(p, method, config=config)
    outputs = model(pixel_values, mask=bool_masked_pos, reconstruction_mode=reconstruction_mode)
    return outputs, bool_masked_pos


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_video(image_list, save_path=None, override_save=False):
    fig, ax = plt.subplots()

    # Function to update the plot with each frame
    def update(frame):
        ax.clear()
        frame_to_show = image_list[frame]
        #
        # ax.imshow(np.transpose(frame_to_show, [1,2,0]))
        ax.imshow(frame_to_show)
        ax.axis('off')

    # Create an animation
    ani = FuncAnimation(fig, update, frames=len(image_list), repeat=True)

    # Save the animation
    if save_path is not None:
        # create folders if they don't exist
        create_directory_if_not_exists(directory=osp.dirname(save_path))

        # check if a file with the same name exists
        if osp.exists(save_path) and (not override_save):
            raise f'{save_path} allready exists, set override to true to override'

        # get writer name
        file_type = os.path.splitext(save_path)[-1][1:]
        if file_type == 'gif':
            writer = 'imagemagick'
        elif file_type == 'mp4':
            writer = 'ffmpeg'
        else:
            raise f'Save video type must gif or mp4 but it was {file_type} instead'
        # save the file
        print(f'Saving file {save_path}')
        ani.save(save_path, writer=writer, fps=30)  # Adjust the fps as needed

    # Display the animation as a GIF in the Jupyter Notebook
    display(HTML(ani.to_jshtml()))

    # Close the figure to avoid a double display
    plt.close()


def transform_video(video):
    # pixel_values_raw = feature_extractor(video, return_tensors="pt").pixel_values
    # pixel_values = rearrange(pixel_values_raw, 'b t c h w -> b c t h w')

    pixel_values_raw = data_transform(video)
    pixel_values = rearrange(pixel_values_raw, 'F C H W -> 1 F C H W')

    return pixel_values


def post_processing(outputs, mask, pixel_values, image_std, image_mean, reconstruction_mode):
    outputs = outputs.detach().numpy()
    try:
        mask = mask.detach().numpy()
    except:
        pass
    image_std_torch = np.array(image_std)[None, :, None, None, None]
    image_mean_torch = np.array(image_mean)[None, :, None, None, None]
    pixel_values_unnorm = pixel_values * image_std_torch + image_mean_torch
    videos_patch = rearrange(pixel_values_unnorm, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=16,
                             p2=16)

    output_mean = 0.48
    output_std = 0.08
    outputs_unnorm = outputs * output_std + output_mean

    if mask is None:
        outputs_reconstruction = outputs_unnorm
    else:
        B, num_patches = mask.shape
        _, _, patch_pix = outputs.shape  # 3 c * 16 p * 16 p
        outputs_reconstruction = np.zeros((B, num_patches, patch_pix))

        if reconstruction_mode:
            num_mask = np.sum(~mask)
            outputs_reconstruction[~mask, :] = outputs_unnorm[:, :num_mask, :]
            outputs_reconstruction[mask, :] = outputs_unnorm[:, num_mask:, :]
        else:
            outputs_reconstruction[~mask, :] = videos_patch[~mask, :]
            outputs_reconstruction[mask, :] = outputs_unnorm

    # outputs_reconstruction[:,:] = outputs_unnorm
    patch_size = 16
    video_reconstruction = rearrange(outputs_reconstruction, 'b (t h w) (p0 p1 p2 c) -> b (t p0) c (h p1) (w p2)', t=8,
                                     h=14, w=14, p0=2, p1=patch_size, p2=patch_size)
    video_reconstruction = video_reconstruction.squeeze()
    video_reconstruction_transposed = video_reconstruction.transpose(0, 2, 3, 1)
    return video_reconstruction_transposed


def reconstruct_video(model, video, mask_prob=0.5, mask_method='bottom', reconstruction_mode=False, save_path=None,
                      override_save=False, args=None, image_std=IMG_STD, image_mean=IMG_MEAN):
    '''
    Run inference on a video and recreate the video and saves it as an mp4 or gif file
    '''
    pixel_values = transform_video(video)
    mask = get_mask(p=mask_prob, method=mask_method, config=args)
    outputs = model(pixel_values, mask=mask, reconstruction_mode=reconstruction_mode)

    video_reconstruction_transposed = post_processing(outputs, mask, pixel_values, image_std, image_mean,
                                                      reconstruction_mode)

    plot_video(video_reconstruction_transposed, save_path=save_path, override_save=override_save)


def load_model(model_path, model_type=None):
    if model_type is None:
        model_type = pretrain_videomae_small_patch16_224

    model = model_type(decoder_depth=4)
    checkpoint = torch.load(
        model_path, map_location="cpu"
    )
    model.load_state_dict(checkpoint['model'])
    return model


def get_name_from_path(path):
    return osp.split(path)[-1].split('.')[0]


if __name__ == '__main__':
    # model_hf, feature_extractor = load_model_hf()
    # model_path = '/home/ubuntu/efs/trained_models/lsfb_isol_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1364.pth'
    # model_path = '/home/ubuntu/efs/trained_models/lsfb_isol_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1414.pth'
    # model_path = '/home/ubuntu/efs/trained_models/lsfb_isol_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1569.pth'
    model_path = '/home/ubuntu/efs/videoMAE/pretrained/VideoMAE_ViT-S_checkpoint_Kinetics-400.pth'
    # model_path = '/home/ubuntu/efs/videoMAE/pretrained/VideoMAE_ViT-B_checkpoint_Kinetics-400.pth'
    # model_path = '/home/ubuntu/efs/videoMAE/pretrained/VideoMAE_ViT-S_checkpoint_ssv2.pth'
    # model_path = '/home/ubuntu/efs/trained_models/Kinetics-400_finetune_ted_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e400/checkpoint-400.pth'
    # model_path = '/home/ubuntu/efs/videoMAE/pretrained/VideoMAE _ViT-H_checkpoint_Kinetics-400.pth'
    model = pretrain_videomae_small_patch16_224(decoder_depth=4)
    # model = pretrain_videomae_base_patch16_224(decoder_depth=4)
    # model = pretrain_videomae_huge_patch16_224()
    checkpoint = torch.load(
        model_path, map_location="cpu"
    )
    model.load_state_dict(checkpoint['model'])

    data_transform = video_transforms.Compose([
        video_transforms.Resize(244, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(224, 224)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMG_MEAN,
                                   std=IMG_STD)
    ])

    # Load video
    root_folder = '/data/lsfb_dataset'
    all_videos_paths = glob.glob(osp.join(root_folder, 'isol', 'videos', '*.mp4'))
    video_path = all_videos_paths[1]
    video = load_video(video_path)
    video = [a.numpy() for a in video]

    image_std, image_mean = IMG_STD, IMG_MEAN

    mask_prob = 0.5
    mask_method = 'horizontal'
    override_save = False
    save_path = None
    reconstruction_mode = True
    args = None

    pixel_values = transform_video(video)
    mask = get_mask(p=mask_prob, method=mask_method, config=args)
    outputs = model(pixel_values, mask=mask, reconstruction_mode=reconstruction_mode)
    video_reconstruction_transposed = post_processing(outputs, mask, pixel_values, image_std, image_mean,
                                                      reconstruction_mode)
    plot_video(video_reconstruction_transposed, save_path=save_path, override_save=override_save)
    #
    # reconstruct_video(model, [a.numpy() for a in video], mask_prob=0.99, mask_method='horizontal', save_path=None,
    #                   reconstruction_mode=True)
