# Set the path to save checkpoints
# OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/eval_lr_5e-4_repeated_aug_epoch_30'
OUTPUT_DIR='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_5vid_clean/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_kinetic_400'

# path to SSV2 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='YOUR_PATH/list_ssv2'
DATA_PATH='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_5vid_clean/dataset'


# path to pretrain model
# MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-2399.pth'
# MODEL_PATH='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_5vid/checkpoint-2400.pth'

POS_WEIGHTS_PATH='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_5vid_clean/dataset/weights.json'

NUM_TRAINERS=8
MODEL_PATH='/home/ubuntu/efs/videoMAE/pretrained/VideoMAE_ViT-B_checkpoint_Kinetics-400.pth'

#/opt/conda/envs/pytorch/bin/python3\
#OMP_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1\
#NCCL_DEBUG=INFO\
OMP_NUM_THREADS=1 \
# python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1 --use-env\
# python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1 \

 /opt/conda/envs/pytorch/bin/torchrun \
  --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set dyadic_communication \
    --multi_labels \
    --pos_weight_path ${POS_WEIGHTS_PATH}\
    --nb_classes 15 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 500 \
    --dist_eval \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --enable_deepspeed \
    --mixup 0 \
    --cutmix 0 \
    --mixup_prob 0 \
    --smoothing 0 \



