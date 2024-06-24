from types import SimpleNamespace as Namespace

import sys
import os
import os.path as osp

# Get the current script's directory
current_dir = osp.dirname(osp.realpath(__file__))

# Navigate two levels up to get the root folder
root_folder = osp.abspath(osp.join(current_dir, '..', '..','..'))

# Add the root folder to the Python path
sys.path.append(root_folder)

from run_class_finetuning import main

# Set the path to save checkpoints
# OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/eval_lr_5e-4_repeated_aug_epoch_30'
OUTPUT_DIR=r'D:\Project-mpg microgesture\human_micro_gesture_classifier\scripts\testing'

# path to SSV2 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='YOUR_PATH/list_ssv2'
DATA_PATH=r'D:\Project-mpg microgesture\human_micro_gesture_classifier\scripts\mac\mac_multi\dataset'

# path to pretrain model
# MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-2399.pth'
MODEL_PATH = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\mac_multi\checkpoint-9.pth'
DATA_ROOT = r'D:\Project-mpg microgesture\mac2024\track1'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)

args = Namespace(

    #multilabel weights
    multi_labels = False,
    one_hot_labels = True,
    hierarchical_labels = True,
    pos_weight_path = osp.join(DATA_PATH,'weights.json'),
    data_root = DATA_ROOT,
    limit_data = 16,

    batch_size= 2,
    short_side_size= 224 ,
   
    update_freq=1,

    # Model parameters
    model='vit_base_patch16_224',
    tubelet_size = 2,
    input_size= 224 ,

    fc_drop_rate=0.0,
    drop=0.0,
    attn_drop_rate=0.0,
    drop_path=0.1,
    disable_eval_during_finetuning=False,
    model_ema=False,
    model_ema_decay=0.9999,
    model_ema_force_cpu=False,

    # Optimizer parameters
    opt= 'adamw',
    opt_eps=1e-8,
    opt_betas= [0.9, 0.999],
    clip_grad=None,
    momentum=0.9,
    weight_decay= 0.05 ,
    weight_decay_end=None,

    lr= 5e-4 ,
    layer_decay=0.75,
    warmup_lr=1e-6,
    min_lr=1e-6,

    warmup_epochs=5,
    warmup_steps=-1,

    epochs= 30 ,
    test_num_segment= 2 ,
    test_num_crop= 3 ,
    dist_on_itp=False,

    # Random Erase params
    reprob = 0.25,
    remode='pixel',
    recount=1,
    resplit=False,    

    # Mixup params
    # mixup=0.8,
    # cutmix = 1.0,
    # cutmix_minmax = None,   
    # mixup_prob=1.0,
    # mixup_switch_prob=0.5,
    # mixup_mode='batch',
    
    mixup=0,
    cutmix = 0,
    cutmix_minmax = None,   
    mixup_prob=0,
    mixup_switch_prob=0,
    mixup_mode=None,
    

    # Augmentation parameters
    color_jittert=0.4,     
    num_sample=1,
    aa='rand-m7-n4-mstd0.5-inc1',
    smoothing=0,
    train_interpolation='bicubic',          

     # Finetuning params
    finetune= MODEL_PATH,
    model_key='model|module',
    model_prefix='',
    init_scale=0.001,
    use_checkpoint=False,
    use_mean_pooling=True,

        # Dataset parameters
    data_path= DATA_PATH,
    eval_data_path = DATA_PATH,
    nb_classes= 56,
    imagenet_default_mean_and_std = True, 
    num_frames= 16 ,
    num_segments = 1,
    sampling_rate = 4,
    data_set = 'dyadic_communication_mpigroup',
    log_dir= OUTPUT_DIR,
    output_dir= OUTPUT_DIR,
    device = 'cuda',
    seed = 42,
    
    auto_resume=True,
    resume='',
    # parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    


    save_ckpt_freq=10 ,
    save_ckpt=True,


    start_epoch=0,
    eval=True,
    dist_eval=False,
    num_workers = 10,
    pin_mem = True,
    enable_deepspeed=False,




)
if __name__ == '__main__':
    main(args, None)