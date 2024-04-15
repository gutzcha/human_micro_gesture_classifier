import os
import torch
import os.path as osp
from types import SimpleNamespace as Namespace


import sys


# Get the current script's directory
current_dir = osp.dirname(osp.realpath(__file__))

# Navigate two levels up to get the root folder
root_folder = osp.abspath(osp.join(current_dir, '..', '..','..'))

# Add the root folder to the Python path
sys.path.append(root_folder)

from utils import load_matching_state_dict
from modeling_finetune import vit_base_patch16_224



args = Namespace(multi_labels=True, pos_weight_path='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/dataset/weights.json', batch_size=6, epochs=30, update_freq=1, save_ckpt_freq=10, model='vit_base_patch16_224', tubelet_size=2, input_size=224, fc_drop_rate=0.0, drop=0.0, attn_drop_rate=0.0, drop_path=0.1, disable_eval_during_finetuning=False, model_ema=False, model_ema_decay=0.9999, model_ema_force_cpu=False, opt='adamw', opt_eps=1e-08, opt_betas=[0.9, 0.999], clip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=None, lr=0.0005, layer_decay=0.75, warmup_lr=1e-06, min_lr=1e-06, warmup_epochs=5, warmup_steps=-1, color_jitter=0.4, num_sample=2, aa='rand-m7-n4-mstd0.5-inc1', smoothing=0.0, train_interpolation='bicubic', crop_pct=None, short_side_size=224, test_num_segment=2, test_num_crop=3, reprob=0.25, remode='pixel', recount=1, resplit=False, mixup=0.0, cutmix=0.0, cutmix_minmax=None, mixup_prob=0.0, mixup_switch_prob=0.5, mixup_mode='batch', finetune='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/checkpoint-2399.pth', model_key='model|module', model_prefix='', init_scale=0.001, use_checkpoint=False, use_mean_pooling=True, data_path='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/dataset', eval_data_path=None, nb_classes=12, imagenet_default_mean_and_std=True, num_segments=1, num_frames=16, sampling_rate=4, data_set='dyadic_communication', output_dir='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400', log_dir='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400', device='cuda', seed=0, resume='', auto_resume=True, save_ckpt=True, start_epoch=0, eval=False, dist_eval=True, num_workers=10, pin_mem=True, world_size=8, local_rank=-1, dist_on_itp=False, dist_url='env://', enable_deepspeed=True, deepspeed=False, deepspeed_config='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/deepspeed_config.json', deepscale=False, deepscale_config=None, deepspeed_mpi=False, rank=0, gpu=0, distributed=True, dist_backend='nccl')
# generate the model
model = vit_base_patch16_224(args)

checkpoint_path  = '/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/checkpoint-2399.pth'

state_dict = torch.load(checkpoint_path)['module']

# load weights 
load_matching_state_dict(model, state_dict)

to_save = {'model': model.state_dict()}

# save model
save_path = '/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_001/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/checkpoint-2400.pth'
torch.save(to_save, save_path )
