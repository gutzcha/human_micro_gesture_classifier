import imageio
import os
import glob
import argparse

def split_video_frames(input_video_path, output_folder, x_offset=0, overwrite=False):
    
    left_output_folder = os.path.join(output_folder,'left')
    if not os.path.exists(left_output_folder):
        os.makedirs(left_output_folder)
    right_output_folder = os.path.join(output_folder,'right')
    if not os.path.exists(right_output_folder):
        os.makedirs(right_output_folder)


    # Open the video file
    input_video = imageio.get_reader(input_video_path)

    video_file_name = os.path.basename(input_video_path)

    # Get video properties
    fps = input_video.get_meta_data()['fps']
    # width, height = input_video.get_meta_data()['size']

    # Create VideoWriter objects for the two output videos
    video_left_path = os.path.join(left_output_folder, video_file_name)
    video_right_path = os.path.join(right_output_folder, video_file_name)

    if not os.path.exists(video_left_path) or not os.path.exists(video_right_path) or overwrite:
            

        video_left = imageio.get_writer(video_left_path, fps=fps, macro_block_size=None)
        video_right = imageio.get_writer(video_right_path, fps=fps, macro_block_size=None)

        for frame_number, frame in enumerate(input_video):
            # Split the frame in the middle
            mid = frame.shape[1] // 2
            left_frame = frame[:, :mid-x_offset, :]
            right_frame = frame[:, x_offset+mid:, :]

            # Write frames to the output videos
            video_left.append_data(left_frame)
            video_right.append_data(right_frame)

        # Close the output videos
        video_left.close()
        video_right.close()

        print(f"Videos saved to {output_folder}")
    # else:
    #     print(f"Videos already exist in {output_folder}")

def main(input_path, output_folder, suffix, folder_level, overwrite=False):
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
    for input_video in file_list:
         # Extract the name of the folder from the file path
        # folder_name = os.path.basename(os.path.dirname(input_video))
        folder_name = "/".join(input_video.split('/')[-(folder_level):-1])

        # Add suffix to the folder name
        new_folder_name = folder_name + suffix

        if output_folder is not None:
            new_file_path = os.path.join(output_folder, new_folder_name) 
        
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)

        
        split_video_frames(input_video, new_file_path, overwrite=overwrite)

def parse_args():

    parser = argparse.ArgumentParser(description="Split video files into segments.")
    parser.add_argument("input_video", help="File path, list of file paths, or folder path for the source of the video files to split.")
    parser.add_argument("--output_folder", help="Folder path for the output. If not provided, new folders will be created in the same location as the input videos.")
    parser.add_argument("--suffix", default="_test", help="Suffix to be added at the end of the newly created folders. If empty or None, output_folder can't be None.")

    return parser.parse_args()


if __name__ == "__main__":
    # input_video_path = "/videos/mpi_data/2Itzik/dyadic_communication/PIS_ID_000_SPLIT/Cam3_split/0000.mp4"
    # output_folder = "/videos/mpi_data/2Itzik/dyadic_communication/PIS_ID_000_SPLIT/test"

    input_video_path = "/videos/mpi_data/2Itzik/dyadic_communication/SEGMENTED"
    output_folder = "/videos/mpi_data/2Itzik/dyadic_communication/SPLIT"
    suffix = ''
    print('Splitting files left and right')
    main(input_video_path, output_folder, suffix,folder_level=3, overwrite=False)

    # args = parse_args()

    # # Validate and process the arguments
    # input_video = args.input_video
    # output_folder = args.output_folder
    # suffix = args.suffix

    # if output_folder is None and (suffix is None or suffix == ''):
    #     raise ValueError("If output_folder is None, suffix can't be empty or None.")

    # main(input_video, output_folder, suffix)
    

