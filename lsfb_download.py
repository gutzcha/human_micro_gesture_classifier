import os
import os.path as osp
from lsfb_dataset import Downloader

dest_isol = osp.join('/data_1000','lsfb_dataset','isol')
dest_cont = osp.join('/data_1000','lsfb_dataset','cont')

os.makedirs(dest_isol, exist_ok=True)
os.makedirs(dest_cont, exist_ok=True)

print('Now downloading isolated files')
print('Now downloading isolated files - skipping existing files')

downloader = Downloader(
    dataset='isol', destination=dest_isol, skip_existing_files=True, include_videos=True, include_raw_poses=False, include_cleaned_poses=False)
downloader.download()
print('Finished downloading isolated files')
print('Now downloading cont files')
downloader = Downloader(
    dataset='cont', destination=dest_cont, skip_existing_files=True, include_videos=True, include_raw_poses=False, include_cleaned_poses=False)
downloader.download()
print('Finished downloading cont files')