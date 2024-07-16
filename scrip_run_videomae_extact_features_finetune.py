from run_videomae_vis_folder_extreact_features import main
from types import SimpleNamespace as Namespace
import os.path as osp
import os
from glob import glob

# image_path_list = glob(osp.join(samples_folder,'**','*.mp4'))
samples_folder = r'D:\Project-mpg microgesture\mac2024\track1\val'
image_path_list = glob(osp.join(samples_folder, '**', '*.mp4'))

models_dict_list = [
    {
        'experiment': 'mac_multi',
        'description': '',
        'checkpoint_path': r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\mac_multi\split_loss_fine_coarse_70\checkpoint-9.pth',
        'model_name': 'vit_base_patch16_224',
    },
]

# Generate save folders
save_folder_root = r"D:\Project-mpg microgesture\human_micro_gesture_classifier\extreacted_features_split_loss_fine_coarse_70"

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
        save_path=save_folder,  # list
        model_path=model_path,
        mask_type='tube',
        num_frames=16,
        sampling_rate=4,
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
