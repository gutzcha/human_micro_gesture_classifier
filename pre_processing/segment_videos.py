import csv
import subprocess
import os.path as osp
import os
import glob
import argparse

def split_video_segment(input_video, output_folder, segment_length):
    # Run ffmpeg command to split the video into segments

    #Get frame numbers
    # frame_rate = 30.001
    # vid_n_frames = 
    # frames_to_extract = range(0,)

    subprocess.call(f'ffmpeg -i {input_video} -c:v libx264 -crf 22 -map 0 -segment_time {segment_length} -g {segment_length} -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*{segment_length})" -reset_timestamps 1 -f segment {output_folder}%4d.mp4', shell=True)


def split_video_from_csv(input_video, output_folder, begin_timestamp, end_timestamp):
    # Create the output filename based on the timestamps
    output_filename = f"{begin_timestamp}_{end_timestamp}.mp4"
    output_path = output_folder + "/" + output_filename

    # Check if the file already exists, if yes, skip
    try:
        with open(output_path, 'r'):
            print(f"Skipping existing file: {output_filename}")
            return
    except FileNotFoundError:
        pass

    # Run ffmpeg command to extract the segment
    subprocess.run([
        'ffmpeg',
        '-i', input_video,
        '-ss', str(begin_timestamp / 1000),  # begin timestamp in seconds
        '-to', str(end_timestamp / 1000),    # end timestamp in seconds
        '-c', 'copy',
        output_path
    ])

def main_from_csv(input_video, output_folder, annotation_csv):

    # Read CSV file with timestamps
    with open(annotation_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            begin_timestamp = int(row['begin'])
            end_timestamp = int(row['end'])

            # Split the video based on timestamps
            split_video(input_video, output_folder, begin_timestamp, end_timestamp)

def main(input_path, output_folder, segment_length, suffix='_split',folder_level=1):

    assert folder_level>=1, "folder_level must be at least 1"
    # If input is a single file name, convert it to a list
    if isinstance(input_path, str):
        file_list = [input_path]
    
    # If input is a list of file names, continue
    if isinstance(input_path, list):
        file_list = input_path
    else:
        # If input is a folder path, get all .mp4 files
        if os.path.isdir(input_path):
            file_list = glob.glob(os.path.join(input_path, '**', '*.mp4'), recursive=True)
        else:
            raise ValueError("Invalid input. Please provide a valid file name, a list of file names, or a folder path.")
    
    # Continue processing with the file_list
    for input_video in file_list[2:]:
        
        # Extract the name of the folder from the file path
        folder_name = "/".join(input_video.split('/')[-(folder_level):-1])
        # folder_name = os.path.basename(os.path.dirname(input_video))
        
        # Add suffix to the folder name
        new_folder_name = folder_name + suffix

        if output_folder is not None:
            new_file_path = os.path.join(output_folder, new_folder_name) 
        
        os.makedirs(new_file_path, exist_ok=True)
        full_file_path = osp.join(new_file_path,osp.basename(input_video))[:-4]

        # Split the video into segments
        split_video_segment(input_video, full_file_path, segment_length)


def parse_args():
    parser = argparse.ArgumentParser(description="Split video files into segments.")
    parser.add_argument("input_video", help="File path, list of file paths, or folder path for the source of the video files to split.")
    parser.add_argument("--output_folder", help="Folder path for the output. If not provided, new folders will be created in the same location as the input videos.")
    parser.add_argument("--segment_length", type=float, default=2, help="Length of each segment to split in seconds. Default is 2.")
    parser.add_argument("--suffix", default="_test", help="Suffix to be added at the end of the newly created folders. If empty or None, output_folder can't be None.")

    return parser.parse_args()

if __name__ == "__main__":
    main_folder = "/videos/mpi_data/2Itzik/dyadic_communication"
    # input_video=osp.join(main_folder, "PIS_ID_000/Cam1/PIS_ID_00_2_Cam1_20200811_043527.036.mp4")
    # output_folder=osp.join(main_folder, "PIS_ID_000/Cam1_split_test/")
    
    # os.makedirs(output_folder, exist_ok=True)


    input_video=osp.join(main_folder,"RAW")
    output_folder = osp.join(main_folder, "SEGMENTED")
    # annotation_csv=osp.join(main_folder, "annotations.csv")
    segment_length = 2

    main(input_video, output_folder,segment_length, suffix='', folder_level=3)



    # args = parse_args()

    # # Validate and process the arguments
    # input_video = args.input_video
    # output_folder = args.output_folder
    # segment_length = args.segment_length
    # suffix = args.suffix

    # if output_folder is None and (suffix is None or suffix == ''):
    #     raise ValueError("If output_folder is None, suffix can't be empty or None.")

    # main(input_video, output_folder, segment_length, suffix)

