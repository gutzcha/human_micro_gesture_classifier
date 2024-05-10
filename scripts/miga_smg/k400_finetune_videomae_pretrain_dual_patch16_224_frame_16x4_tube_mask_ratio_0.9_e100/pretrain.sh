# Set the path to save checkpoints

EXPERIMENT_NAME='miga_smg'
RUN_NAME='k400_finetune_videomae_pretrain_dual_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100'
OUTPUT_DIR_ROOT='/fast/ygoussha/pretrained'

OUTPUT_DIR=${OUTPUT_DIR_ROOT}/${EXPERIMENT_NAME}/${RUN_NAME}
# Set the path to MPIIGroupInteraction train set. 
DATA_PATH='/code/scripts/'${EXPERIMENT_NAME}/${RUN_NAME}'/train.txt'

# batch_size can be adjusted according to number of GPUs
# this script is for 8 GPUs (1 nodes x 8 GPUs)
export OMP_NUM_THREADS=1

python /lustre/home/ygoussha/human_micro_gesture_classifier/run_mae_pretraining_multi_decoder.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224_densepose_dual \
        --decoder_depth 4 \
        --batch_size 2 \
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
        --root='/fast/ygoussha/miga_challenge/smg_split_files/'

