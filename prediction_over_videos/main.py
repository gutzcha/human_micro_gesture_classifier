import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, VideoLoader, cpu, gpu
import numpy as np
from torchvision import transforms
from run_videomae_vis_v2 import DataAugmentationForVideoMAEInference, save_list_of_images_as_video
from types import SimpleNamespace as Namespace
import os
from timm.models import create_model
from utils import load_state_dict
from typing import List, Union
import pandas as pd
from scipy.special import softmax,  expit
from tqdm import tqdm

try:
    from accelerate import Accelerator

    accelerator = Accelerator()
except:
    Warning('Accelerator not installed, using single GPU')
    accelerator = None

from mpigroup.load_model_inference import get_args

def softmax_func(x):
    return softmax(x, axis=1)
def get_transformation():
    args = Namespace(
        mask_type=None,
        input_size=224)

    return DataAugmentationForVideoMAEInference(args)


def reshape_results(results: np.ndarray, num_dataset) -> np.ndarray:
    # reshape
    # logits b_r, classes -> b, v, classes
    batch_res, classes = results.shape
    batch_size = batch_res // num_dataset
    return results.reshape(batch_size, num_dataset, classes)


def smooth_results(results: np.ndarray, smooth_func=None) -> np.ndarray:
    # use softmax or sigmoid to smooth the logits before aggregation
    if smooth_func is not None:
        results = smooth_func(results)
    return results


def aggregate_resolutions(logits):
    return logits.mean(axis=1)


def process_predictions(logits, window_len_list, frame_inds, labels, smooth_func=None):
    # reshape results b*v, c, t, h, w -> b, v, c, t, h, w
    logits = reshape_results(logits, num_dataset=len(window_len_list))

    batch_size = logits.shape[0]

    # smooth results use softmax or sigmoid
    logits_smoothed = smooth_results(logits, smooth_func=smooth_func)

    # aggregate results
    logits_aggregated = aggregate_resolutions(logits)
    logits_smoothed_aggregated = aggregate_resolutions(logits_smoothed)

    # get predictions
    predictions = np.argmax(logits_smoothed.reshape(-1, logits_smoothed.shape[-1]), axis=1,
                            ).reshape(logits_smoothed.shape[0], logits_smoothed.shape[1])
    predictions_aggregated = np.argmax(logits_smoothed_aggregated, axis=1)

    # summarize predictions
    preds = []
    for batch_ind in range(batch_size):
        frame_ind = frame_inds[batch_ind]
        preds_temp = {
            'frame_ind': frame_ind,
            'resolution': 'aggregated',
            'logits': [logits_aggregated[batch_ind]],
            'smoothed_logits': [logits_smoothed_aggregated[batch_ind]],
            'predictions': [predictions_aggregated[batch_ind]],
            'labels': [labels[batch_ind]]
        }
        preds.append(pd.DataFrame.from_dict(preds_temp))
        for ind, window_len in enumerate(window_len_list):
            preds_temp = {
                'frame_ind': frame_ind,
                'resolution': [window_len],
                'logits': [logits[batch_ind, ind]],
                'smoothed_logits': [logits_smoothed[batch_ind, ind]],
                'predictions': [predictions[batch_ind, ind]],
                'labels': [labels[batch_ind]]
            }
            preds.append(pd.DataFrame.from_dict(preds_temp))

    return pd.concat(preds)

class Results():
    def __init__(self, log_path):
        self.log_path = log_path

    def get_header(self):
        pass

class VideoDataset(Dataset):
    def __init__(self, video_path, num_frames=16, window_length=16, step_length=1,
                 transformations=None, video_loader=None, start_frame=None, end_frame=None):
        self.video_path = video_path
        self.num_frames = num_frames
        self.window_length = window_length
        self.step_length = step_length
        self.transformations = transformations

        self.video_loader = VideoReader(video_path, ctx=cpu()) if video_loader is None else video_loader
        self.video = self.video_loader

        self.frame_ratio = self.window_length // self.num_frames

        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else len(self.video_loader) - 1
        self.video_length = self.end_frame - self.start_frame + 1
        self.real_video_length = len(self.video_loader)

        self.current_frame = 0

    def get_frame_inds(self, index):
        start_frame = (index * self.step_length) + self.start_frame - self.window_length // 2  # centralize the index
        end_frame = start_frame + self.window_length
        inds = np.arange(start_frame, end_frame, self.frame_ratio)
        # handle edge cases: start and end of video
        inds[inds < 0] = 0
        inds[inds >= self.real_video_length] = self.real_video_length - 1

        return inds

    def __len__(self):
        # return max(0, ((self.video_length - self.window_length) // self.step_length) + 1)
        # i am padding the video so no need to subtract the window size
        return max(0, ((self.video_length + 1) // self.step_length) + 1)

    def __getitem__(self, index):
        # index = index_in + self.start_frame  # shift the start of the video
        frames_inds = self.get_frame_inds(index)
        frames = self.video.get_batch(frames_inds).asnumpy()
        frames = [
            transforms.ToPILImage()(frame) for frame in frames
        ]
        if self.transformations:
            frames, _ = self.transformations((frames, None))
            frames = frames.view((self.num_frames, 3) + frames.size()[-2:]).transpose(0, 1)
        return frames


class MultiTimeResVideoDataset(Dataset):
    def __init__(self, dataset_list, labels_path=None):

        self.dataset_list: List[VideoDataset] = dataset_list
        self.window_length_list = [d.window_length for d in dataset_list]
        self.labels_path = labels_path
        self.labels_df = self.read_csv(labels_path)
        self.start_frame = dataset_list[0].start_frame

    def read_csv(self, file_path):
        if file_path is not None:
            df = pd.read_csv(file_path, header=None)
            # Add headers to the columns
            df.columns = ["label", "start_frame", "end_frame"]
        else:
            df = None
        return df

    def get_labels_from_csv(self, index):
        if self.labels_df is None:
            return -1
        actual_index = index + self.start_frame
        for _, row in self.labels_df.iterrows():
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            label = row['label']

            if start_frame <= actual_index <= end_frame:
                return label

        # If the index is not contained within any frame range, return -1
        return -1

    def __len__(self):
        return self.dataset_list[0].__len__()

    def __getitem__(self, index):
        tensor_list = [dataset[index] for dataset in self.dataset_list]
        label = self.get_labels_from_csv(index)
        return torch.stack(tensor_list, dim=0), index, label


class MultiTimeResVideoInference:
    def __init__(self, args):
        self.device: torch.device = torch.device(args.device)  # this can also be accelerator.device
        self.model: torch.nn.Module = self._create_model(args)
        self.args = self.add_model_arguments_to_args(args)
        self.num_frames: int = args.num_frames
        self.step_length: int = args.step_length
        self.window_length_list: List[int] = args.window_length_list
        self.video_path: str = args.video_path
        self.video_loader: VideoReader = VideoReader(self.video_path, ctx=cpu())


        self.start_frame: int = args.start_frame if args.start_frame is not None else 0
        self.end_frame: int = args.end_frame if args.end_frame is not None else self.video_length - 1
        self.video_length: int = self.end_frame - self.end_frame + 1

        self.transformations = args.transformations
        self.datasets: List[VideoDataset] = self.get_datasets()
        self.num_dataset = len(self.datasets)
        self.multi_dataset = MultiTimeResVideoDataset(self.datasets)
        self.batch_size: int = args.batch_size
        self.dataloader = DataLoader(self.multi_dataset, batch_size=self.batch_size, shuffle=False)
        self.smooth_function = args.smooth_function
        self.log_path = args.log_path
        self.labels_path = args.labels_path

        if accelerator is not None:
            self.device = accelerator.device
            self.model, self.dataloader = accelerator(self.model, self.dataloader)

    def predict(self, log_path=None):
        model = self.model
        model.to(self.device)
        model.eval()
        log_path = self.log_path if log_path is None else log_path
        all_preds = pd.DataFrame()
        window_len_list = self.window_length_list
        with torch.no_grad():
            for batch, frame_inds, labels in tqdm(self.dataloader, total=(len(self.dataloader))):
                batch = batch.reshape(-1, *batch.shape[2:])
                batch = batch.to(self.device)

                logits = model(batch)
                logits = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                frame_inds = frame_inds.detach().cpu().numpy().tolist()

                batch_sum = process_predictions(logits, window_len_list, frame_inds, labels, self.smooth_function)
                # Write results to CSV file
                # with open(log_path, 'a') as f:
                #     batch_sum.to_csv(f, header=f.tell() == 0, index=False)
                batch_sum.to_csv(log_path, mode='a', index=False, header=False) #(log_path, mode=’a’, index=False, header=False)
                all_preds = pd.concat([all_preds, batch_sum], ignore_index=True)

        return all_preds

    def get_datasets(self):

        datasets = [VideoDataset(video_path=self.video_path,
                                 num_frames=self.num_frames,
                                 window_length=i,
                                 step_length=self.step_length,
                                 video_loader=self.video_loader,
                                 start_frame=self.start_frame,
                                 end_frame=self.end_frame,
                                 transformations=self.transformations

                                 ) for i in self.window_length_list]
        return datasets

    def add_model_arguments_to_args(self, args):
        patch_size = self.model.patch_embed.patch_size
        args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
        args.patch_size = patch_size
        return args

    def _create_model(self, args):
        # load model
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_checkpoint=args.use_checkpoint,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )

        checkpoint = torch.load(args.finetune, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            checkpoint_model = checkpoint['module']

        load_state_dict(model, checkpoint_model)
        model = model.to(self.device)
        model.eval()
        return model


def test_process_predictions():
    batch_size = 2
    n_views = 3
    n_classes = 17

    logits = np.random.randn(batch_size * n_views, n_classes)
    window_len_list = [[16, 32, 64]] * batch_size
    frame_inds = list(range(batch_size))
    targets = list(range(batch_size))

    outputs = process_predictions(logits, window_len_list, frame_inds, targets)
    print(outputs.shape)


def test_dataset():
    # Example usage:
    # Define transformations
    args = Namespace(
        mask_type=None,
        num_frames=16,
        sampling_rate=4,
        input_size=224,
        densepose=True)

    transform = DataAugmentationForVideoMAEInference(args)

    # Initialize dataset
    video_path = r'D:\Project-mpg microgesture\smg\SMG_RGB_Phase2\smg_rgb_test\Sample0039\Sample0039_color.mp4'
    datasets = [VideoDataset(video_path=video_path, num_frames=16, window_length=i, step_length=1,
                             transformations=transform) for i in [16, 32, 64]]
    multi_dataset = MultiTimeResVideoDataset(dataset_list=datasets)

    batch_size = 2
    multi_dataset_dataloader = DataLoader(multi_dataset, batch_size=batch_size, shuffle=False)
    multi_inference_iter = iter(multi_dataset_dataloader)

    batch, frame_inds, targets = next(multi_inference_iter)
    window_len_list = [16, 32, 64]

    n_views = len(window_len_list)
    n_classes = 17

    logits = np.random.randn(batch_size * n_views, n_classes)
    frame_inds = frame_inds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    outputs = process_predictions(logits, window_len_list, frame_inds, targets)
    # sample = multi_dataset[0]
    #
    # print(sample.shape)

    # fps_orig = 30
    # # Access a sample
    # for dataset in datasets:
    #     frame_ind = 64
    #     # dataset_len = len(dataset)
    #     # print(f'{dataset_len=}')
    #     sample = dataset[frame_ind]
    #     window_length = dataset.window_length
    #     filename = os.path.basename(video_path)
    #     full_name = f'{frame_ind}_{window_length}_{filename}'
    #     fps = max(fps_orig // dataset.frame_ratio, 1)
    #     save_list_of_images_as_video(image_array_list=sample, output_path=full_name, fps=fps, unnormalize=True)


def test_MultiTimeResVideoInference(video_path, model_path, num_frames=16, sampling_rate=None,
                                    step_length=1, window_lengths=None, start_frame=None, end_frame=None,
                                    labels_path=None):
    # video_path, model_path, num_frames, sampling_rate, step_length, window_lengths
    yaml_path = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\model_configs\miga_smi.yaml'
    args_namespase, all_params_dict = get_args(yaml_path)
    args_namespase.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # args_namespase.num_segments = 1
    args_namespase.video_path = video_path
    args_namespase.model_path = model_path
    args_namespase.num_frames = num_frames
    args_namespase.sampling_rate = sampling_rate
    args_namespase.step_length = step_length
    args_namespase.window_length_list = window_lengths if window_lengths is not None else [32]
    args_namespase.start_frame = start_frame
    args_namespase.end_frame = end_frame
    args_namespase.smooth_function = softmax_func
    args_namespase.labels_path = labels_path
    args_namespase.transformations = get_transformation()

    add_start_end = f'_{start_frame:04}_{end_frame:04}' if start_frame is not None else ''
    video_name = os.path.basename(args_namespase.video_path).split('.')[0]
    out_log_path = f'prediction_over_videos_{video_name}{add_start_end}_new_resampled.csv'

    args_namespase.log_path = out_log_path

    predictor = MultiTimeResVideoInference(args_namespase)

    predictor.predict(out_log_path)


if __name__ == "__main__":
    # test_dataset()
    # test_process_predictions()
    video_path = r'D:\Project-mpg microgesture\smg\SMG_RGB_Phase1\smg_rgb_validate\Sample0031\Sample0031_color.mp4'
    # model_path = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\retrained_base_model_CondensedNearestNeighbour'
    # model_path = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\resampled_64_multi_update_freq_8\checkpoint-best.pth'
    model_path = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\resampled_64_multi_update_freq_8\down_sampled_data\checkpoint-best.pth'
    labels_path = r'D:\Project-mpg microgesture\smg\smg_data_phase1\smg_skeleton_validate\Sample0031\Sample0031_finelabels.csv'
    start_frame = 12000
    end_frame = 14000
    test_MultiTimeResVideoInference(video_path, model_path, num_frames=16, sampling_rate=None,
                                    step_length=4, window_lengths=None, start_frame=start_frame, end_frame=end_frame,
                                    labels_path=labels_path)
