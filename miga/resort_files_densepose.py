import os
import shutil

root_folder = '/lustre/fast/fast/ygoussha/miga_challenge/smg_split_files'

# Create new folders 'densepose_train' and 'densepose_validation'
densepose_train_folder = os.path.join(root_folder, 'densepose_train')
densepose_validation_folder = os.path.join(root_folder, 'densepose_validation')

print('creating densepose_train_folder')
os.makedirs(densepose_train_folder, exist_ok=True)
os.makedirs(densepose_validation_folder, exist_ok=True)

# Transfer files with "densepose" in their names to the respective folders
print('=========================================')
print('transferring files...')
for folder_name in ['train']:
# for folder_name in ['train', 'validation']:
    source_folder = os.path.join(root_folder, folder_name)
    target_folder = densepose_train_folder if folder_name == 'train' else densepose_validation_folder
    n_files = os.listdir(source_folder)
    for ind, filename in enumerate(os.listdir(source_folder)):
        if 'densepose' in filename:
            shutil.move(os.path.join(source_folder, filename), os.path.join(target_folder, filename))

            # Rename files replacing 'densepose' with 'color'
            new_filename = filename.replace('densepose', 'color')
            os.rename(os.path.join(target_folder, filename), os.path.join(target_folder, new_filename))
        if ind%10 == 0:
            print(f'file {ind} of {n_files}')
# Save list of file names without extension to 'train.txt'
print('=========================================')
print('saving filen names')
with open(os.path.join(root_folder, 'train_new.txt'), 'w') as f:
    for filename in os.listdir(densepose_train_folder):
        f.write(os.path.splitext(filename)[0] + '\n')