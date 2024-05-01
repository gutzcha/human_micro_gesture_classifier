from types import SimpleNamespace as Namespace
import os.path as osp
import os
from run_videomae_vis_v2 import main
from timm.models import create_model
import torch
import run_videomae_vis_v2 as rec_tools
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

def get_model(args, load_flag=True):
    device = torch.device(args.device)
    model = rec_tools.get_model(args)

    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    if load_flag:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()

    return model, device

def get_frames(video_path, transformations, n_frames=16, limit_frames=None, skip_frames=4):
    with open(video_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))

    len_vr = len(vr)

    if limit_frames is not None:
        len_vr = min(len_vr, limit_frames)

    frame_id_list = list(range(0, len_vr, skip_frames))

    len_frames = len(frame_id_list)
    if len_frames > n_frames:
        len_frames = (len_frames//n_frames)*n_frames
        frame_id_list = frame_id_list[:len_frames+1]
    else:
        for _ in range(len_frames - n_frames):
            frame_id_list.append(frame_id_list[-1])

    video_data = vr.get_batch(frame_id_list).asnumpy()
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
    img, bool_masked_pos = transformations((img, None))
    img = img.reshape((-1, n_frames, 3) + img.size()[-2:]).transpose(1, 2)
    return img
def run_inference(model,device, video_path, args, limit_frames):
    # get frames
    transformations = rec_tools.DataAugmentationForVideoMAEInference(args)
    img = get_frames(video_path, transformations, n_frames=16, limit_frames=limit_frames, skip_frames=4)
    img = img.to(device, non_blocking=True)
    outputs = []
    with torch.no_grad():
        for seq in img:
            outputs.append(model(seq.unsqueeze(0)))
    return outputs

def main(args):
    model, device = get_model(args)
    video_path = image_path_list[0]
    outputs = run_inference(model, device, video_path, args,limit_frames=256)



if __name__ == '__main__':
    image_path_list = [r'D:\Project-mpg microgesture\smg\SMG_RGB_Phase1\smg_rgb_train\Sample0001\Sample0001_color.mp4']
    image_path_list = [a.replace('\\', '\\\\') for a in image_path_list]

    models_dict = {
        'experiment': 'MPIG_densepose_dual_2',
        'description': 'MPIG_densepose_dual - videoMAE-K400 , same as K400 but then was finetuned on MPIGroupInteractions dataset (train set) for 100 epochs, with denspose as additional decoding target',
        'checkpoint_path': r'D:\Project-mpg microgesture\pretrained\pretrained\MPIIGroupInteraction\k400_finetune_videomae_pretrain_dual_2_patch16_224_frame_16x4_tube_mask_ratio_0.9_e100\checkpoint-99.pth',
        'model_name': 'pretrain_videomae_base_patch16_224_densepose_dual',
    }

    model_path = models_dict['checkpoint_path']
    model_path = model_path.replace('\\', '\\\\')

    path_to_sub = r'D:\Project-mpg microgesture\smg'
    path_to_sub = path_to_sub.replace('\\', '\\\\')


    save_folder = osp.join(path_to_sub, 'reconstructed')
    model = models_dict['model_name']

    # save_folder_root = ''
    #
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

    # model = get_model(args, load_flag=False)

    # video_path = image_path_list[0]
    # run_inference(model, video_path, args)
    main(args)