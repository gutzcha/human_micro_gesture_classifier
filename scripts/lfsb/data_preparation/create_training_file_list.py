import os.path as osp
import pandas as pd
import os
from moviepy.editor import VideoFileClip
from tqdm import tqdm


# Function to get video duration using moviepy
def get_video_duration(file_path):
    try:
        video_clip = VideoFileClip(file_path)
        duration = video_clip.duration
        return duration
    except Exception as e:
        return 0.0  # Return 0 for any error


# Function to process a folder and create the output file
def process_folder(input_folder, info_df, output_file):
    with open(output_file, 'w') as output:
        for _, (fname, label, signer, start_frame, end_frame, label_num) in tqdm(info_df.iterrows(), total=len(info_df)):
            # print(fname, label, signer,start_frame,end_frame)

            video_path = os.path.join(input_folder, fname + '.mp4')
            video_duration = get_video_duration(video_path)

            output.write(f"{video_path} {label_num} {video_duration} {label} {signer} {start_frame} {end_frame}\n")


def replace_and_save_text_file(file_path):
    try:
        # Open and read the file
        with open(file_path, 'r') as file:
            file_contents = file.read()

        # Replace all occurrences of "/data_1000" with "/data"
        modified_contents = file_contents.replace("/data_1000", "/data")

        # Write the modified contents back to the file
        with open(file_path, 'w') as file:
            file.write(modified_contents)

        print(f"File '{file_path}' modified and saved successfully.")

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    root_folder = osp.join('/data', 'lsfb_dataset')
    # instances_file_path = osp.join(root_folder, 'isol', 'instances.csv')
    # instances_df = pd.read_csv(instances_file_path)
    #
    # # recode the signs and generates a map
    # # Create a mapping of unique signs to unique integers
    # unique_signs = instances_df['sign'].unique()
    # sign_to_id = {sign: i for i, sign in enumerate(unique_signs)}
    #
    # # Add the 'sign_id' column based on the mapping
    # instances_df['sign_id'] = instances_df['sign'].map(sign_to_id)
    # print(f'Number of unique signs: {len(unique_signs)}')
    #
    # input_folder = osp.join(root_folder, 'isol', 'videos')  # Replace with the path to your input folder
    output_file = osp.join(root_folder, 'isol', 'train.txt')  # Replace with the desired output file name
    #
    # process_folder(input_folder, instances_df, output_file)

    # Example usage:
    file_path = output_file  # Replace with the actual file path
    replace_and_save_text_file(file_path)
