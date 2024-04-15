import os
import shutil
from glob import glob

DATASETS = ['train','val','test']
def inverse_folders(root,dest_path, folder_to_effect=['openface','clips','openpose']):
    if dest_path is None:
        dest_path = root
    
    found_folder = False
    files = glob(os.path.join(root,'*_test','**'))

    for filename in files:
        # file_parts = os.path.split(filename)
        file_path = os.path.join(root, filename)
        # base_file_name = os.path.splitext(filename)[0]
        # file_parts = filename.split('/')
        for dataset_name in DATASETS:
            if (dataset_name in filename):
                found_folder = True
                break
            else:
                found_folder = False
        if not found_folder:
            continue
        else:
            found_folder = False
        # dataset_name = file_parts[-1]
        # Create folder for filename without extension
        feature_name = filename.split('/')[-2].split('_')[0]
        base_file_name = filename.split('/')[-1].split('.')[0]
        new_folder_path = os.path.join(dest_path, dataset_name, base_file_name,feature_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Move the file into the new folder
        shutil.move(file_path, new_folder_path)

# directory_path = '/Users/itzikg/workspace/mpi_data/2Itzik/MPIIGroupInteraction'
# new_dirname = '/Users/itzikg/workspace/mpi_data/2Itzik/MPIIGroupInteraction/features'
directory_path = '/home/ubuntu/data_local/mpi_data/2Itzik/MPIIGroupInteraction'
new_dirname = '/home/ubuntu/data_local/mpi_data/2Itzik/MPIIGroupInteraction/features'            

inverse_folders(directory_path,new_dirname, ['openface','video','openpose'])