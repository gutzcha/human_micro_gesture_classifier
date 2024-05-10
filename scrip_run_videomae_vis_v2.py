from run_videomae_vis_v2 import main
# from run_videomae_vis_folder import main
# from run_videomae_vis_folder_extreact_features import main
from types import SimpleNamespace as Namespace
import os.path as osp
import os
from glob import glob

# samples_folder = '/home/ubuntu/efs/video_samples'
# samples_folder = '/home/ubuntu/efs/video_samples'
# Folder structure:
# dataset name
# - dataset type (train, val)
#   - file name.mp4

# When saving:
# experiment/epoch/dataset name/ dataset type/ file name/ modality (images/videos) / files
# files-images: ori_image, rec_image, ori_image_dense, rec_image_dense
# files-videos: ori_video, rec_video, ori_video_dense, rec_video_dense


# image_path_list = glob(osp.join(samples_folder,'**','*.mp4'))

# video_folder_name = '/videos/mpi_data/2Itzik/MPIIGroupInteraction/clips_train'
# video_folder_name_val = '/videos/mpi_data/2Itzik/MPIIGroupInteraction/clips_val'
# video_folder_name2 = '/videos/k400/train'
video_folder_name3 = 'videos\\smg'
# image_path_list = [
#     osp.join(video_folder_name_val,'06245-video1.mp4'),
#     osp.join(video_folder_name_val,'07012-video.mp4'),
#     osp.join(video_folder_name_val,'07778-video2.mp4'),
#     osp.join(video_folder_name_val,'38137-video1.mp4'),
#     osp.join(video_folder_name_val,'38904-video.mp4'),
#     osp.join(video_folder_name,'00000-video.mp4'),
#     osp.join(video_folder_name,'00000-video1.mp4'),
#     osp.join(video_folder_name,'00000-video2.mp4'),
#     osp.join(video_folder_name,'00001-video.mp4'),
#     osp.join(video_folder_name,'00001-video1.mp4'),
#     osp.join(video_folder_name2,'Fka21pOT9UE_000179_000189.mp4'),
#     osp.join(video_folder_name2,'F6dHpIG2vL8_000002_000012.mp4'),
#     osp.join(video_folder_name2,'Mwreo2lMhcI_000002_000012.mp4'),
#     osp.join(video_folder_name2,'Uhyx80-ZyQE_000045_000055.mp4'),
#     osp.join(video_folder_name2,'zzzzE0ncP1Y_000232_000242.mp4')
#     ]
image_path_list = [
    # osp.join(video_folder_name3, "Sample0031_color0000.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0001.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0002.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0003.mp4"),
    osp.join(video_folder_name3, "Sample0031_color0010.mp4"),
    osp.join(video_folder_name3, "Sample0031_color0011.mp4"),
    osp.join(video_folder_name3, "Sample0031_color0012.mp4"),
    osp.join(video_folder_name3, "Sample0031_color0013.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0064.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0065.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0066.mp4"),
    # osp.join(video_folder_name3, "Sample0031_color0067.mp4"),
]

# We have several different models, each trained with a different setting:
# 1. K400 - videoMAE trained on k400 for 1600 epochs, this is the baseline we are trying to improve
# 2. MPIG_base - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs
# 3. MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target
# 4. MPIG_densepose_singal - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with only denspoise as decoding target
# All model architectures are based on pretrain_videomae_base_patch16_224, MPIG_densepose is also based on that model with the addition of an additional decoder for densepose


models_dict_list = [
#     {
#     'experiment':'K400',
#     'description':'K400 - videoMAE trained on k400 for 1600 epochs, this is the baseline we are trying to improve',
#     'checkpoint_path':'',
#     'model_name':'pretrain_videomae_base_patch16_224',
# },
#     {
#     'experiment':'MPIG_base',
#     'description':'MPIG_base - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs',
#     'checkpoint_path':[],
#     'model_name':'pretrain_videomae_base_patch16_224',
# },
#     {
#     'experiment':'MPIG_densepose_dual',
#     'description':'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
#     'checkpoint_path':'/videos/pretrained/MPIIGroupInteraction/k400_finetune_videomae_pretrain_dual_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100/checkpoint-99.pth',
#     'model_name':'pretrain_videomae_base_patch16_224_densepose_dual',
# },
#     {
#     'experiment':'MPIG_densepose_singal',
#     'description':'MPIG_densepose_singal - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with only denspoise as decoding target',
#     'checkpoint_path':[],
#     'model_name':'pretrain_videomae_base_patch16_224',
# },
#     {
#     'experiment':'MPIG_densepose_dual_fresh_decoder',
#     'description':'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
#     'checkpoint_path':'/videos/pretrained/MPIIGroupInteraction/k400_finetune_videomae_pretrain_dual_fresh_decoder_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100/checkpoint-19.pth',
#     'model_name':'pretrain_videomae_base_patch16_224_densepose_dual',
# },
#        {
#     'experiment':'MPIG_densepose_dual_2',
#     'description':'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
#     'checkpoint_path':'/videos/pretrained/MPIIGroupInteraction/k400_finetune_videomae_pretrain_dual_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100/checkpoint-69.pth',
#     'model_name':'pretrain_videomae_base_patch16_224_densepose_dual',
# },

#        {
#     'experiment':'MPIG_densepose_dual_2',
#     'description':'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
#     'checkpoint_path':'/videos/pretrained/MPIIGroupInteraction/k400_finetune_videomae_pretrain_dual_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100/checkpoint-99.pth',
#     'model_name':'pretrain_videomae_base_patch16_224_densepose_dual',
# },
       {
    'experiment':'MPIG_densepose_dual_2',
    'description':'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
    'checkpoint_path':r'D:\Project-mpg microgesture\pretrained\pretrained\MPIIGroupInteraction\k400_finetune_videomae_pretrain_dual_2_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100\checkpoint-99.pth',
    'model_name':'pretrain_videomae_base_patch16_224_densepose_dual',
},


]

# Generate save folders

# save_folder_root = '/home/ubuntu/efs/videoMAE/video_samples_results'
save_folder_root = r"D:\Project-mpg microgesture\human_micro_gesture_classifier\video_samples_results"

save_path_list = []
for models_dict in models_dict_list:
    experiment = models_dict['experiment']
    checkpoint_name = models_dict['checkpoint_path'].split('/')[-1].split('.')[0]
    save_folder = osp.join(save_folder_root, experiment, checkpoint_name)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    save_path_list.append(save_folder)

for models_dict, save_folder in zip(models_dict_list, save_path_list):
    model_path = models_dict['checkpoint_path']
    model = models_dict['model_name']
    # try:
    args = Namespace(
            img_path=image_path_list,
            save_path=save_folder, # list
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
            densepose=True,
            drop_path=0.0)
    main(args=args)
    # except:
        # pass
