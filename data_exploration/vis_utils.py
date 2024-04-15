from run_videomae_vis_v2 import reconstruct_video_from_patches, save_video

def reconstruct_and_save_video(target_input,bool_masked_pos,model_output,
                               video_save_path):
    imgs, rec_imgs, mask = reconstruct_video_from_patches(
    ori_img=target_input,
    patch_size=[16,16],
    bool_masked_pos=bool_masked_pos,
    outputs=model_output,
    frame_id_list=None
    )

    save_video(ori_img=rec_imgs, 
               video_save_path=f'reconstructed_{video_save_path}',
               frame_id_list=None, 
               prepend='reconstructed', 
               vid_ext='mp4',
               fps=7,
               )
    
    save_video(ori_img=target_input, 
               video_save_path=f'original_{video_save_path}',
               frame_id_list=None, 
               prepend='original', 
               vid_ext='mp4',
               fps=7,
               )

    