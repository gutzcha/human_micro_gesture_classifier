import os.path as osp
import pandas as pd
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from glob import glob
import concurrent.futures


# Function to get video duration using moviepy
def get_video_duration(file_path):
    try:
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        return duration
    except Exception as e:
        return 0.0  # Return 0 for any error

# Function to process a single video file and return its duration
def process_video(video_path):
    return (video_path, get_video_duration(video_path))


# Function to process a folder and create the output file using multiple processes
def process_folder(file_names, output_file):
    with open(output_file, 'w') as output:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Process each video file concurrently and write results to the output file
            results = list(tqdm(executor.map(process_video, file_names), total=len(file_names)))
            for video_path, video_duration in results:
                output.write(f"{video_path} {video_duration} \n")

if __name__ == '__main__':
    root_folder = '/videos/mpi_data/2Itzik/MPIIGroupInteraction/clips_train'
    file_names = glob(osp.join(root_folder, '*.mp4'))
    experiment_folder = '/home/ubuntu/efs/videoMAE/scripts/MPIIGroupInteraction/k400_finetune_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/'
    output_file = osp.join(experiment_folder,'train.txt')  # Replace with the desired output file name

    process_folder(file_names, output_file)