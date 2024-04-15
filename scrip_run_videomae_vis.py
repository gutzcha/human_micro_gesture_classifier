from run_videomae_vis import main
# from run_videomae_vis_folder import main
# from run_videomae_vis_folder_extreact_features import main
from types import SimpleNamespace as Namespace
import os.path as osp
import os

video_folder_name = '/videos/mpi_data/2Itzik/MPIIGroupInteraction/clips_train'
image_paths = [
    osp.join(video_folder_name,'00000-video.mp4'),
    osp.join(video_folder_name,'00000-video1.mp4'),
    osp.join(video_folder_name,'00000-video2.mp4'),
    osp.join(video_folder_name,'00001-video.mp4'),
    osp.join(video_folder_name,'00001-video1.mp4'),
]

OUTPUT_DIR_ROOT='/videos/pretrained'
EXPERIMENT_NAME='MPIIGroupInteraction'
RUN_NAME='k400_finetune_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100'
OUTPUT_DIR=osp.join(OUTPUT_DIR_ROOT,EXPERIMENT_NAME,RUN_NAME)

# baseline
# checkpoint_name = 'VideoMAE_ViT-B_checkpoint_Kinetics-400.pth'
# model_path = osp.join('/videos/pretrained/pretrained',checkpoint_name)

# this run
# checkpoint_name = 'checkpoint-19.pth'
checkpoint_name = 'checkpoint-100.pth'

model_path = osp.join(OUTPUT_DIR, checkpoint_name)

model = 'pretrain_videomae_base_patch16_224'
save_path = osp.join(OUTPUT_DIR,'reconstructed',checkpoint_name)
os.makedirs(save_path, exist_ok=True)
for image_path in image_paths:
    # try:
    args = Namespace(img_path=image_path,
            save_path=save_path,
            model_path=model_path,
            mask_type='tube',
            num_frames=16,
            sampling_rate=4,
            decoder_depth=4,
            input_size=224,
            device='cuda:0',
            imagenet_default_mean_and_std=True,
            mask_ratio=0,
            model=model,
            drop_path=0.0)
    main(args=args)
    # except:
        # pass
    