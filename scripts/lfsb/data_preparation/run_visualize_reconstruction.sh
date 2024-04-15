#opts    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
#    parser.add_argument('img_path', type=str, help='input video path')
#    parser.add_argument('save_path', type=str, help='save video path')
#    parser.add_argument('model_path', type=str, help='checkpoint path of model')
#    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
#                        type=str, help='masked strategy of video tokens/patches')
#    parser.add_argument('--num_frames', type=int, default= 16)
#    parser.add_argument('--sampling_rate', type=int, default= 4)
#    parser.add_argument('--decoder_depth', default=4, type=int,
#                        help='depth of decoder')
#    parser.add_argument('--input_size', default=224, type=int,
#                        help='videos input size for backbone')
#    parser.add_argument('--device', default='cuda:0',
#                        help='device to use for training / testing')
#    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
#    parser.add_argument('--mask_ratio', default=0.75, type=float,
#                        help='ratio of the visual tokens/patches need be masked')
#    # Model parameters
#    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
#                        help='Name of model to vis')
#    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
#                        help='Drop path rate (default: 0.1)')
#
#    return parser.parse_args()


#
## Set the path to save checkpoints
#OUTPUT_DIR='/home/ubuntu/efs/trained_models/Kinetics-400_finetune_ted_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e400'
## Set the path to Kinetics train set.
#DATA_PATH='/data/lsfb_dataset/isol/train.txt'
#NUM_TRAINERS=8
#
##/opt/conda/envs/pytorch/bin/python3\
##OMP_NUM_THREADS=1 /opt/conda/envs/pytorch/bin/python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1\
##NCCL_DEBUG=INFO\
#
## python3 -m torch.distributed.launch --nproc-per-node=8 --nnodes=1 --use-env\
#OMP_NUM_THREADS=1\
# /opt/conda/envs/pytorch/bin/torchrun \
#  --standalone\
#    --nnodes=1\
#    --nproc-per-node=$NUM_TRAINERS\
#           run_mae_pretraining.py\
#            --data_path ${DATA_PATH} \
#            --mask_type tube \
#            --mask_ratio 0.9 \
#            --model pretrain_videomae_small_patch16_224 \
#            --decoder_depth 4 \
#            --batch_size 64 \
#            --num_frames 16 \
#            --sampling_rate 4 \
#            --opt adamw \
#            --opt_betas 0.9 0.95 \
#            --warmup_epochs 40 \
#            --save_ckpt_freq 50 \
#            --epochs 401 \
#            --log_dir ${OUTPUT_DIR} \
#            --output_dir ${OUTPU}

#CLSFBI2104A_S045_B_368276_368707.mp4

/opt/conda/envs/pytorch/bin/python3 run_videomae_vis.py \
--img_path '/data/lsfb_dataset/cont/videos/CLSFBI0103A_S001_B.mp4' \
--save_path '/data/lsfb_dataset/reconstructed' \
--model_path '/data/videoMAE/trained_models/Kinetics-400_finetune_lsfb_isol_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e400/checkpoint-400.pth' \
--mask_type 'tube' \
--num_frames 16 \
--sampling_rate 4 \
--decoder_depth 4 \
--input_size 224 \
--device 'cuda:0' \
--mask_ratio 0.75 \
--model pretrain_videomae_small_patch16_224
