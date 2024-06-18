import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from decord import VideoReader

print(cv2.__version__)
from glob import glob


def get_indices(path_to_file, path_to_labels, n_frames):
    # load csv file and get variables
    df = pd.read_csv(path_to_labels, header=None)
    df.columns = ['labels', 'start_frame', 'end_frame']
    labels_list = df['labels'].values
    start_frame_list = df['start_frame'].values
    end_frame_list = df['end_frame'].values
    duration_list = end_frame_list - start_frame_list

    n_labels = len(labels_list)

    # subtracts 1 from labels to shift 1-17 -> 0-16
    labels_list = labels_list - 1

    # combine frame getter
    list_of_label_lists = []
    list_of_onehots = []
    list_of_frame_lists = []
    new_start_inds = []
    new_end_inds = []
    list_of_file_names = []

    i = 0
    duration_offset = duration_list[i] // 2
    if duration_offset > WIN_LEN // 2:
        duration_offset = WIN_LEN // 2  # start from the first frame

    # get first ind
    start_frame = max([start_frame_list[i] + duration_offset - WIN_LEN // 2, 0])
    end_frame = min([start_frame + WIN_LEN, n_frames])
    j = 0
    i = 0
    while i < n_labels - 1:

        # init labels
        new_labels_list = [labels_list[i]]

        # init onehot
        onehot = np.zeros(N_CLASSES)
        onehot[labels_list[i]] = 1

        # check if there is a new label to append
        k = i + 1
        while k < n_labels and start_frame_list[k] < end_frame:
            # add that labels to the list of the current ONLY if this is new
            if labels_list[k] not in new_labels_list:
                new_labels_list.append(labels_list[k])
                onehot[labels_list[k]] = 1
            k += 1
        list_of_label_lists.append(new_labels_list)
        list_of_onehots.append(onehot)
        this_frames_list = list(range(start_frame, end_frame))
        if len(this_frames_list) < WIN_LEN:
            # just ignore the last 2 seconds
            # this_frames_list += [this_frames_list[-1]]*(WIN_LEN - len(this_frames_list))
            break

        this_file_name_base = os.path.basename(path_to_file).split('.')[0]

        new_start_inds.append(start_frame)
        new_end_inds.append(end_frame)
        list_of_frame_lists.append(this_frames_list)

        # if the video was longer than 64 frames, splitting until finished
        if end_frame < end_frame_list[i]:
            start_frame = end_frame
            end_frame = min([end_frame + WIN_LEN, n_frames])
        else:
            i += 1
            duration_offset = duration_list[i] // 2
            if duration_offset > WIN_LEN // 2:
                duration_offset = WIN_LEN // 2  # start from the first frame

            # get first ind
            start_frame = max([start_frame_list[i] + duration_offset - WIN_LEN // 2, 0])
            end_frame = min([start_frame + WIN_LEN, n_frames])

        labels_string = '-'.join([str(int(i)) for i in new_labels_list])
        this_file_name = f'{this_file_name_base}_{j:03}_{start_frame:05}-{end_frame:05}_{labels_string}.mp4'
        this_file_name = this_file_name.replace('Sample', '')
        this_file_name = this_file_name.replace('color_', '')

        list_of_file_names.append(this_file_name)
        j += 1
    # combine everything into a dataframe
    df_out = pd.DataFrame(zip(list_of_file_names,
                              new_start_inds,
                              new_end_inds,
                              list_of_label_lists,
                              list_of_onehots,
                              list_of_frame_lists
                              ), columns=['file_name', 'start_frame', 'end_frame', 'labels', 'multihot_labels',
                                          'frame_numbers'])
    # add folder name and full path name
    df_out['source_folder'] = os.path.dirname(path_to_file)
    df_out['source_video'] = os.path.basename(path_to_file)
    return df_out


def save_video(output_video_path, frames, fps=30):
    if not frames:
        print("Error: No frames to write.")
        return

    # Determine the frame size from the first frame
    height, width, _ = frames[0].shape

    # Check if output directory exists, create if not
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as per your requirement
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height))  # Convert fps to float

    if not out.isOpened():
        print("Error: Failed to open VideoWriter.")
        return

    try:
        # Write frames to the video
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
    except Exception as e:
        print("Error occurred during video writing:", str(e))
    finally:
        # Release the VideoWriter object
        out.release()
        # print("Video writing completed.")


def load_frames(path_to_file, frame_numbers):
    vr = cv2.VideoCapture(path_to_file)
    if not vr.isOpened():
        print("Error: Unable to open video file.")
        return
    frames = []
    for frame_number in frame_numbers:
        vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = vr.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Error reading frame {frame_number} from video {path_to_file}")
    vr.release()
    return frames


# Save the files and labels
def save_files(df_in, path_to_file, fps, output_video_folder):
    try:
        vr = VideoReader(path_to_file)

        for _, row in tqdm(df_in.iterrows(), total=len(df_in)):

            file_name = row['file_name']
            output_video_path = os.path.join(output_video_folder, file_name)

            if os.path.exists(output_video_path):
                # make sure that it has 64 frames
                # debug
                # vr_temp = VideoReader(output_video_path)
                # shape = vr_temp.get_batch([1]).asnumpy().shape
                # shape_ref = (1,1080,1920,3)
                # if shape != shape_ref:
                #     print(f"Error: Video {file_name} shape: {shape}")
                #
                continue


            frame_numbers = row['frame_numbers']
            # print('Reading file')
            # frames = load_frames(path_to_file, frame_numbers)
            frames = vr.get_batch(frame_numbers).asnumpy()
            frames = [a for a in frames]

            # print('Saving file')
            save_video(output_video_path, frames, fps)

    except Exception as e:
        print("An error occurred:", str(e))


def get_video_details(path_to_file):
    cap = cv2.VideoCapture(path_to_file)
    length, fps, height, width = [], [], [], []
    if not cap.isOpened():
        print("could not open :", path_to_file)
        return
    try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        cap.release()
        return length, fps, height, width


def process_one_video(path_to_file, path_to_labels, output_video_path, output_csv_path):
    length, fps, height, width = get_video_details(path_to_file)

    df_out = get_indices(path_to_file, path_to_labels, length)
    save_files(df_out, path_to_file, fps, output_video_path)
    csv_name = os.path.basename(path_to_file).replace('mp4', 'csv')
    df_out.to_csv(os.path.join(output_csv_path, csv_name))


def process_all_videos(folder_path_all, path_to_labels_all, output_video_path, output_csv_path):
    n_files = len(folder_path_all)
    print('Processing {} videos'.format(n_files))
    for i, (path_to_file, path_to_labels) in enumerate(zip(folder_path_all, path_to_labels_all)):
        print('=====================================')
        print(f'=== file {i} of {n_files} ======')
        process_one_video(path_to_file, path_to_labels, output_video_path, output_csv_path)


def test_one_file():
    path_to_file = r"C:\Users\gutzc\GitHub\human_micro_gesture_classifier\miga_dataset\testing split files\Sample0001_color.mp4"
    path_to_labels = r"C:\Users\gutzc\GitHub\human_micro_gesture_classifier\miga_dataset\testing split files\Sample0001_finelabels.csv"

    length, fps, height, width = get_video_details(path_to_file)

    df_out = get_indices(path_to_file, path_to_labels, length)
    df_out.to_csv(os.path.join(os.path.dirname(path_to_file), 'testing.csv'))
    # save_files(df_out, path_to_file, fps)
    print(df_out)


if __name__ == '__main__':
    WIN_LEN = 64
    N_CLASSES = 17
    folder_path_video = r'C:\Users\gutzc\GitHub\human_micro_gesture_classifier\miga_dataset\SMG_RGB_Phase1'
    folder_path_labels = r'C:\Users\gutzc\GitHub\human_micro_gesture_classifier\miga_dataset\smg_data_phase1'

    path_to_all_videos = glob(os.path.join(folder_path_video, '*', '*', '*.mp4'))


    path_to_all_labels = glob(os.path.join(folder_path_labels, '*', '*', '*finelabels.csv'))


    # remove already split files
    path_to_all_videos = [a for a in path_to_all_videos if 'split_files_video' not in a]
    path_to_all_labels = [a for a in path_to_all_labels if 'split_files_video' not in a]

    output_video_path = os.path.join(folder_path_video, 'split_files_video')
    output_csv_path = os.path.join(folder_path_video, 'split_files_csv')

    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_csv_path, exist_ok=True)

    process_all_videos(folder_path_all=path_to_all_videos,
                       path_to_labels_all=path_to_all_labels,
                       output_video_path=output_video_path,
                       output_csv_path=output_csv_path)
