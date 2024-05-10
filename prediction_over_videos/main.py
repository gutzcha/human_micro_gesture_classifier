import torch
from torch.utils.data import Dataset
from decord import VideoReader


class InferVideos(Dataset):

    def __init__(self, video_path, n_frames, sampling_rate, n_overlap_frames, transform=None):
        super(InferVideos, self).__init__()
        self.video_path = video_path
        self.n_frames = n_frames
        self.sampling_rate = sampling_rate
        self.n_overlap_frames = n_overlap_frames
        self.transform = transform

        self.vr = self.get_vr()

    def get_vr(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


