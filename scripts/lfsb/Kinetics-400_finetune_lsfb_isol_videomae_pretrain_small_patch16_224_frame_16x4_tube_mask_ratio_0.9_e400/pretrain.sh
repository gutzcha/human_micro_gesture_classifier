# Set the path to save checkpoints
OUTPUT_DIR='/home/ubuntu/efs/trained_models/Kinetics-400_finetune_ted_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e400'
# Set the path to Kinetics train set. 
DATA_PATH='/data/lsfb_dataset/isol/train.txt'
NUM_TRAINERS=8

#/opt/conda/envs/pytorch/bin/python3\
#OMP_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1\
#NCCL_DEBUG=INFO\

# python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1 --use-env\
OMP_NUM_THREADS=1\
 /opt/conda/envs/pytorch/bin/torchrun \
  --standalone\
    --nnodes=1\
    --nproc-per-node=$NUM_TRAINERS\
           run_mae_pretraining.py\
            --data_path ${DATA_PATH} \
            --mask_type tube \
            --mask_ratio 0.9 \
            --model pretrain_videomae_small_patch16_224 \
            --decoder_depth 4 \
            --batch_size 64 \
            --num_frames 16 \
            --sampling_rate 4 \
            --opt adamw \
            --opt_betas 0.9 0.95 \
            --warmup_epochs 40 \
            --save_ckpt_freq 50 \
            --epochs 401 \
            --log_dir ${OUTPUT_DIR} \
            --output_dir ${OUTPUT_DIR}\
            --world_size 8
