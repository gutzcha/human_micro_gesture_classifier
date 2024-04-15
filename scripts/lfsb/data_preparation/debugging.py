from run_mae_pretraining import main

OUTPUT_DIR = '/home/ubuntu/efs/trained_models/lsfb_isol_videomae_pretrain_small_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600'
DATA_PATH = '/data/lsfb_dataset/isol/train.txt'

args = dict(  data_path=DATA_PATH,
    mask_type='tube', mask_ratio='0.9'
    , model='pretrain_videomae_small_patch16_224'
    , decoder_depth='4'
    , batch_size='64'
    , num_frames='16'
    , sampling_rate='4'
    , opt='adamw'
    , opt_betas='0.9 0.95'
    , warmup_epochs='40'
    , save_ckpt_freq='20'
    , epochs='1601'
    , log_dir=OUTPUT_DIR
    , output_dir=OUTPUT_DIR)
main(args)
