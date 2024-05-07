# Set the path to save checkpoints
# OUTPUT_DIR='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/eval_lr_5e-4_repeated_aug_epoch_30'
ROOT_EXPERIMENT="D:\Project-mpg microgesture\human_micro_gesture_classifier\scripts\miga_smg\videomae_vit_base_patch16_224_kinetic_400_densepose_dual"
OUTPUT_DIR="${ROOT_EXPERIMENT}\outputs"

# path to SSV2 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='YOUR_PATH/list_ssv2'
DATA_PATH="${ROOT_EXPERIMENT}\dataset"

DATA_ROOT="D:\Project-mpg microgesture\smg\smg_split_files"

# path to pretrain model
# MODEL_PATH='YOUR_PATH/ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400/checkpoint-2399.pth'
# MODEL_PATH='/home/ubuntu/efs/videoMAE/scripts/dyadic_communication_5vid/checkpoint-2400.pth'

POS_WEIGHTS_PATH="${DATA_PATH}\weights.json"

NUM_TRAINERS=1
MODEL_PATH="D:\Project-mpg microgesture\pretrained\pretrained\MPIIGroupInteraction\k400_finetune_videomae_pretrain_dual_2_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100\checkpoint-99.pth"

torchrun \
  --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    --run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set dyadic_communication_mpigroup \
    --pos_weight_path "${POS_WEIGHTS_PATH}"\
    --nb_classes 17 \
    --data_path "${DATA_PATH}" \
    --finetune "${MODEL_PATH}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 1 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --dist_eval \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --mixup 0 \
    --cutmix 0 \
    --mixup_prob 0 \
    --smoothing 0 \
    --data_root "${DATA_ROOT}" \
    --one_hot_labels
  #    --multi_labels \

    #    --enable_deepspeed \
