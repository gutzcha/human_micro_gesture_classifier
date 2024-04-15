# Set the path to save checkpoints

EXPERIMENT_NAME='MPIIGroupInteraction'
RUN_NAME='k400_finetune_videomae_pretrain_dual_2_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100'
OUTPUT_DIR_ROOT='/videos/pretrained'

OUTPUT_DIR=${OUTPUT_DIR_ROOT}/${EXPERIMENT_NAME}/${RUN_NAME}
# Set the path to MPIIGroupInteraction train set. 
DATA_PATH='/home/ubuntu/efs/videoMAE/scripts/'${EXPERIMENT_NAME}/${RUN_NAME}'/train.txt'

# batch_size can be adjusted according to number of GPUs
# this script is for 8 GPUs (1 nodes x 8 GPUs)
OMP_NUM_THREADS=1
NUM_TRAINERS=8

/opt/conda/envs/pytorch/bin/torchrun \
        --standalone \
        --nnodes=1 \
        --nproc-per-node=$NUM_TRAINERS \
        run_mae_pretraining_multi_decoder.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224_densepose_dual \
        --decoder_depth 4 \
        --batch_size 8 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 0 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --clone_decoder \
        --root='/home/ubuntu/data_local/MPIIGroupInteraction'

