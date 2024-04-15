import multiprocessing
from pose_utils import PoseCollection
import os
from tqdm import tqdm

# Define a function to process each pose_name
def process_pose(pose_name_dataset):
    pose_name, dataset = pose_name_dataset
    if skip_existing and os.path.exists(os.path.join(root_folder_name, dataset, pose_name, 'heatmap')):
        return
    
    pose_obj_collection = PoseCollection(root_folder_name=root_folder_name,
                                         dataset=dataset,
                                         pose_name=pose_name,
                                         pose_types=['face', 'body_hand'],
                                         feature_folder_name=None,
                                         file_extention=None,
                                         num_joints=None,
                                         image_width=None,
                                         image_height=None,
                                         heat_map_sigma=5,
                                         use_conf_as_sigma=False,
                                         pose_conf_threshold=0.8
                                         )
    pose_obj_collection.save_video()

if __name__ == "__main__":
    
    skip_existing = True

    # Define the home folder and root folder name
    home_folder = '/home/ubuntu'
    root_folder_name = os.path.join(home_folder, 'data_local/mpi_data/2Itzik/MPIIGroupInteraction/features')
    feature_folder_name = None
    dataset_list = ['val','train','test']

    # List of pose names
    # pose_names_list = ['01647-video2', 'other_pose_name1', 'other_pose_name2', ...]
    # Get a list of pose folder names
    pose_names_list = []
    for dataset in dataset_list:
        pose_folder_path = os.path.join(root_folder_name, dataset)
        pose_names_list_temp = [(folder_name, dataset) for folder_name in os.listdir(pose_folder_path) if os.path.isdir(os.path.join(pose_folder_path, folder_name))]
        pose_names_list += pose_names_list_temp





    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Initialize tqdm progress bar with total number of tasks
    with tqdm(total=len(pose_names_list)) as pbar:
        # Define a callback function to update the progress bar
        def update_progress(*_):
            pbar.update()

        # Use tqdm directly with the iterator returned by pool.imap_unordered
        for _ in tqdm(pool.imap_unordered(process_pose, pose_names_list, chunksize=1), total=len(pose_names_list)):
            update_progress()

    # Close the pool to release resources
    pool.close()
    pool.join()