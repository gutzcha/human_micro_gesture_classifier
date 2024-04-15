ROOT_EXPERIMENT="/home/ubuntu/efs/videoMAE/scripts/MPIIGroupInteraction/videomae_vit_base_patch16_224_kinetic_400_base"
OUTPUT_DIR="${ROOT_EXPERIMENT}/outputs"

DATA_PATH="${ROOT_EXPERIMENT}/dataset"

DATA_ROOT="/home/ubuntu/data_local/MPIIGroupInteraction"

POS_WEIGHTS_PATH="${DATA_PATH}/weights.json"

NUM_TRAINERS=8
MODEL_PATH="/videos/pretrained/MPIIGroupInteraction/k400_finetune_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100/checkpoint-100.pth"

OMP_NUM_THREADS=1 \

 /opt/conda/envs/pytorch/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set dyadic_communication_mpigroup \
    --multi_labels \
    --pos_weight_path ${POS_WEIGHTS_PATH} \
    --nb_classes 14 \
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
    --data_root ${DATA_ROOT}



