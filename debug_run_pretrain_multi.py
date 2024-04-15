from datasets import DataAugmentationForVideoMAE
from kinetics import VideoMAE2VID
from types import SimpleNamespace as Namespace
from torch.utils.data import DataLoader 
import torch
from timm.models import create_model
import os.path as osp
import copy
from einops import rearrange
from utils import  clone_decoder_weights
import run_videomae_vis_v2  as vis_utils 
# unnormalize_frames, save_video, reconstruct_video_from_patches
import importlib
from datasets import build_pretraining_dataset_multi_decoder
# importlib.reload(VideoMAE2VID)
import run_mae_pretraining_multi_decoder


# dry test with run_pretrain
checkpoint_name = 'VideoMAE_ViT-B_checkpoint_Kinetics-400.pth'
model_path = osp.join('/videos/pretrained/pretrained',checkpoint_name)

model_name = 'pretrain_videomae_base_patch16_224_densepose_dual'

import argparse

# Create args Namespace with default values
args = argparse.Namespace(
    batch_size=64,
    epochs=800,
    save_ckpt_freq=50,
    decoder_depth=4,
    mask_type='tube',
    mask_ratio=0.75,
    input_size=224,
    drop_path=0.0,
    normlize_target=True,
    opt='adamw',
    opt_eps=1e-8,
    momentum=0.9,
    weight_decay=0.05,
    lr=1.5e-4,
    warmup_lr=1e-6,
    min_lr=1e-5,
    warmup_epochs=40,
    warmup_steps=-1,
    use_checkpoint=False,
    color_jitter=0.0,
    train_interpolation='bicubic',
    data_path='/path/to/list_kinetics-400',
    imagenet_default_mean_and_std=True,
    num_frames=16,
    sampling_rate=4,
    output_dir='',
    log_dir=None,
    device='cuda',
    seed=0,
    auto_resume=True,
    start_epoch=0,
    num_workers=10,
    pin_mem=True,
    world_size=1,
    local_rank=-1,
    dist_on_itp=False,
    dist_url='env://',
    features_cfg=None,
    clone_decoder=False,
    weight_decay_end=None,
    resume='',
    clip_grad=None,
)

# Override default values with values provided in the call from terminal
args.batch_size = 2
args.lr = 0.000003
args.save_ckpt_freq = 20
args.epochs = 500
args.log_dir = 'testing_new_model_debug'
args.output_dir = 'testing_new_model_debug'

args.root = '/videos/mpi_data/2Itzik/MPIIGroupInteraction/'
args.data_path = 'train_data_mpig.txt'
args.mask_type = 'tube' 
args.mask_ratio = 0.9 
args.decoder_depth = 4 
args.batch_size = 2 
args.num_frames = 16 
args.sampling_rate = 4 
args.input_size = 224
args.patch_size = 16
args.window_size = (args.num_frames // 2, args.input_size // args.patch_size, args.input_size // args.patch_size)
args.model_path=model_path
args.device='cuda:0'
args.imagenet_default_mean_and_std=True
args.model=model_name
args.drop_path=0.0
args.clone_decoder = False # True
args.feature_cfg = False



run_mae_pretraining_multi_decoder.main(args)