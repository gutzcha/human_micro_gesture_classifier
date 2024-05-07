import os
import numpy as np
import torch
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms
from run_videomae_vis import DataAugmentationForVideoMAEInference
import ast
import pandas as pd
import re


class DyadicvideoClsDataset(Dataset):

    def __init__(self, anno_path=None, data_path=None, mode='train', clip_len=2,
                 crop_size=224, short_side_size=244, new_height=244,
                 new_width=244, keep_aspect_ratio=True, num_segment=1,
                 num_crop=1, test_num_segment=1, test_num_crop=1,
                 view_crop_mapping=None, corner_crop_size=None, data_root=None, limit_data=None, args=None, **kwargs):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.view_crop_mapping = view_crop_mapping
        self.corner_crop_size = corner_crop_size
        self.data_root = data_root
        self.is_multi_labels = args.multi_labels
        self.is_one_hot_labels = args.one_hot_labels
        self.limit_data = limit_data  # debug 20
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        if isinstance(self.anno_path, str):
            df = pd.read_csv(self.anno_path)
        else:
            df = self.anno_path

        if self.limit_data is not None and self.limit_data <= len(df) and self.limit_data > 0:
            df = df.iloc[:self.limit_data]

        dataset_samples = list(df['filenames'].values)
        if self.data_root is not None:
            if 'folder_name' in df.columns:
                sub_folder = df['folder_name'].values[0]
            else:  # default
                if mode == 'train':
                    sub_folder = 'clips_train'
                elif mode == 'validation':
                    sub_folder = 'clips_val'
                elif mode == 'test':
                    # test can be val in disguise
                    if anno_path.endswith('val.csv'):
                        sub_folder = 'clips_val'
                    else:
                        sub_folder = 'clips_test'
            data_folder = os.path.join(self.data_root, sub_folder)
            dataset_samples = [os.path.join(data_folder, a) for a in dataset_samples]
        self.dataset_samples = dataset_samples
        if self.is_multi_labels or self.is_one_hot_labels:
            self.label_array = list(df['labels'].apply(lambda x: np.array(ast.literal_eval(re.sub(r'[,\s]+', ',', x)))))
        else:
            self.label_array = list(df['labels'])

        # if not is_multi and is_one_hot convert to index
        if not self.is_multi_labels and self.is_one_hot_labels:
            self.label_array = [np.argmax(a) for a in self.label_array]

        self.metadata_array = list(
            df['metadata'].apply(lambda x: np.array(ast.literal_eval(x))))  # convert to list of dicts with metadata
        self.view_list = list(df['view'].values)
        self.data_root = data_root
        # if self.limit_data is not None and self.limit_data <= len(self.dataset_samples) and self.limit_data > 0:
        #     self.dataset_samples = self.dataset_samples[:self.limit_data]
        #     self.view_list = self.view_list[:self.limit_data]
        #     self.metadata_array = self.metadata_array[:self.limit_data]
        #     self.label_array = self.label_array[:self.limit_data]
        if (mode == 'train'):
            pass


        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                # video_transforms.PadToSquare(),
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                # video_transforms.PadToSquare(),
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            view = self.view_list[index]
            metadata = self.metadata_array[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    metadata = self.metadata_array[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            # crop corner
            if self.view_crop_mapping is not None:
                buffer = crop_corner(buffer, self.corner_crop_size, self.view_crop_mapping[view])

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], metadata, [], []

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            view = self.view_list[index]
            metadata = self.metadata_array[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    view = self.view_list[index]
                    metadata = self.metadata_array[index]
                    buffer = self.loadvideo_decord(sample)

            # crop corner
            if self.view_crop_mapping is not None:
                buffer = crop_corner(buffer, self.corner_crop_size, self.view_crop_mapping[view])

            buffer = self.data_transform(buffer)

            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], metadata, [], []

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            view = self.view_list[index]
            metadata = self.metadata_array[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format( \
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                view = self.view_list[index]
                metadata = self.metadata_array[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            # crop corner
            if self.view_crop_mapping is not None:
                buffer = crop_corner(buffer, self.corner_crop_size, self.view_crop_mapping[view])

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            if self.test_num_crop > 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                               / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            else:

                spatial_start = 0
            temporal_start = chunk_nb  # 0/1

            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::2, \
                         spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::2, \
                         :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], metadata, \
                chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
            self,
            buffer,
            args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []
            tick = len(vr) / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                                      [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(
                np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                 size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def crop_corner(frames, crop_size, corner):
    """
    Crop the given frames at the given corner.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        crop_size (int): the size of height and width used to crop the
            frames. if None, then max min of height and width will be used
        corner (str): corner to crop the frames. Valid values are 'tl' (top left),
            'tr' (top right), 'bl' (bottom left), and 'br' (bottom right), and 'mm' (middle point)
    Returns:
        frames (tensor): spatially cropped frames.
    """
    assert corner in ['tl', 'tr', 'bl', 'br', 'mm']
    _, height, width, _ = frames.shape
    if crop_size is None:
        crop_size = min([height, width])
    assert crop_size <= height and crop_size <= width, "Crop size must be smaller than the dimensions of the frames."
    if corner == 'bl':
        frames = frames[:, -crop_size:, :crop_size, :]
    elif corner == 'tr':
        frames = frames[:, :crop_size, -crop_size:, :]
    elif corner == 'br':
        frames = frames[:, -crop_size:, -crop_size:, :]
    elif corner == 'tl':
        frames = frames[:, :crop_size, :crop_size, :]
    else:  # cornet == 'mm'
        crop_height = crop_width = crop_size

        # Calculate crop boundaries
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height
        start_x = (width - crop_width) // 2
        end_x = start_x + crop_width

        frames = frames[:, start_y:end_y, start_x:end_x, :]

    return frames


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
