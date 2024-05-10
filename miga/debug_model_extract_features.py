from types import SimpleNamespace as Namespace

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from run_videomae_vis_dataset_extreact_features import main


args = Namespace(multi_labels=False,
                 pos_weight_path='D:\\Project-mpg microgesture\\human_micro_gesture_classifier\\scripts\\miga_smg\\overfit_videomae_vit_base_patch16_224_kinetic_400_densepose_dual\\dataset\\weights.json',
                 batch_size=1, epochs=20, update_freq=1, save_ckpt_freq=1, model='vit_base_patch16_224',
                 tubelet_size=2, input_size=224, fc_drop_rate=0.0, drop=0.0, attn_drop_rate=0.0,
                 drop_path=0.1, disable_eval_during_finetuning=False,
                 model_ema=False, model_ema_decay=0.9999,
                 model_ema_force_cpu=False, opt='adamw',
                 opt_eps=1e-08, opt_betas=[0.9, 0.999],
                 clip_grad=None, momentum=0.9,
                 weight_decay=0.05, weight_decay_end=None,
                 lr=0.0005, layer_decay=0.75,
                 warmup_lr=1e-06, min_lr=1e-06,
                 warmup_epochs=1, warmup_steps=-1,                 color_jitter=0.4, num_sample=2,
                 aa='rand-m7-n4-mstd0.5-inc1', smoothing=0.0,
                 train_interpolation='bicubic', crop_pct=None,
                 short_side_size=224, test_num_segment=1, test_num_crop=1, reprob=0.25, remode='pixel',
                 recount=1, resplit=False, mixup=0.0, cutmix=0.0, cutmix_minmax=None, mixup_prob=0.0,
                 mixup_switch_prob=0.5, mixup_mode='batch',
                 finetune='D:\\Project-mpg microgesture\\pretrained\\pretrained\\MPIIGroupInteraction\\k400_finetune_videomae_pretrain_dual_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100\\checkpoint-99.pth',
                 model_key='model|module', model_prefix='',
                 init_scale=0.001, use_checkpoint=False, use_mean_pooling=True,
                 data_path='D:\\Project-mpg microgesture\\human_micro_gesture_classifier\\scripts\\miga_smg\\overfit_videomae_vit_base_patch16_224_kinetic_400_densepose_dual\\dataset',
                 eval_data_path=None, nb_classes=17, imagenet_default_mean_and_std=True,
                 num_segments=1, num_frames=16, sampling_rate=4,
                 data_set='dyadic_communication_mpigroup',
                 output_dir='D:\\Project-mpg microgesture\\human_micro_gesture_classifier\\scripts\\miga_smg\\overfit_videomae_vit_base_patch16_224_kinetic_400_densepose_dual\\outputs',
                 data_root='D:\\Project-mpg microgesture\\smg\\smg_split_files',
                 log_dir='D:\\Project-mpg microgesture\\human_micro_gesture_classifier\\scripts\\miga_smg\\overfit_videomae_vit_base_patch16_224_kinetic_400_densepose_dual\\outputs',
                 device='cuda', seed=0, resume='', auto_resume=True, save_ckpt=True, start_epoch=0, eval=False,
                 dist_eval=True, num_workers=10, pin_mem=True, world_size=1, local_rank=-1, dist_on_itp=False,
                 dist_url='env://', enable_deepspeed=False, distributed=False, one_hot_labels=False, limit_data=None)

if __name__ == '__main__':
    main(args, None)