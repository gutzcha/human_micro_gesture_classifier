import yaml
import os
import os.path as osp
from mpigroup.load_model_inference import pars_path, get_args
from run_class_finetuning import main as run_class_finetuning
# from run_class_finetuning import get_args as main_get_args
import argparse


# Example usage
def pars_args(**kwargs):
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification',
                                     add_help=False)

    # Define the default values for optional arguments
    default_values = {
        'multi_labels': False, 'pos_weight_path': '', 'batch_size': 64, 'epochs': 30, 'update_freq': 1,
        'save_ckpt_freq': 100, 'model': 'vit_base_patch16_224', 'tubelet_size': 2, 'input_size': 224,
        'fc_drop_rate': 0.0, 'drop': 0.0, 'attn_drop_rate': 0.0, 'drop_path': 0.1,
        'disable_eval_during_finetuning': False, 'model_ema': False, 'model_ema_decay': 0.9999,
        'model_ema_force_cpu': False, 'opt': 'adamw', 'opt_eps': 1e-8, 'opt_betas': None,
        'clip_grad': None, 'momentum': 0.9, 'weight_decay': 0.05, 'weight_decay_end': None,
        'lr': 1e-3, 'layer_decay': 0.75, 'warmup_lr': 1e-6, 'min_lr': 1e-6,
        'warmup_epochs': 1, 'warmup_steps': -1, 'color_jitter': 0.4, 'num_sample': 2,
        'aa': 'rand-m7-n4-mstd0.5-inc1', 'smoothing': 0.1, 'train_interpolation': 'bicubic',
        'crop_pct': None, 'short_side_size': 224, 'test_num_segment': 5, 'test_num_crop': 3,
        'reprob': 0.25, 'remode': 'pixel', 'recount': 1, 'resplit': False, 'mixup': 0.8,
        'cutmix': 1.0, 'cutmix_minmax': None, 'mixup_prob': 1.0, 'mixup_switch_prob': 0.5,
        'mixup_mode': 'batch', 'finetune': '', 'model_key': 'model|module', 'model_prefix': '',
        'init_scale': 0.001, 'use_checkpoint': False, 'use_mean_pooling': True,
        'data_path': '/path/to/list_kinetics-400',
        'eval_data_path': None, 'nb_classes': 400, 'imagenet_default_mean_and_std': True,
        'num_segments': 1, 'num_frames': 16, 'sampling_rate': 4, 'data_set': 'Kinetics-400',
        'output_dir': '', 'data_root': '', 'log_dir': None, 'device': 'cuda', 'seed': 0,
        'resume': '', 'auto_resume': True, 'save_ckpt': True, 'start_epoch': 0, 'eval': False,
        'dist_eval': False, 'num_workers': 10, 'pin_mem': True, 'world_size': 1,
        'local_rank': -1, 'dist_on_itp': False, 'dist_url': 'env://', 'enable_deepspeed': False
    }

    # Update the default values with provided kwargs
    default_values.update(kwargs)

    # Add arguments to the parser
    for key, value in default_values.items():
        parser.add_argument(f'--{key}', default=value)

    args = parser.parse_args([])  # Parse empty list to avoid parsing terminal arguments
    return args


path_to_json = osp.join("..", "model_configs", "miga_train_debug.yaml")
_, config_dict = get_args(path_to_json, params=None)
config_new = pars_args(**config_dict)
run_class_finetuning(config_new, None)
