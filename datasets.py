import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE, VideoMAE2VID
from ssv2 import SSVideoClsDataset
from dyadic_communication import DyadicvideoClsDataset
from mpigroup.const import cropping_map as mpigroup_cropping_map
from dyadic_communication_densepose import DensposeDyadicvideoClsDataset
# import one_click_dense_pose.utils.
class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_pretraining_dataset_multi_decoder(args):

    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE2VID(
        root=args.root,
        setting=args.data_path,
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        features_cfg=args.features_cfg,
        )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args, anno_path=None):
    if args.data_set == 'Kinetics-400':
        mode = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174
    elif args.data_set == 'dyadic_communication':
        mode = None
        # anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        dataset = DyadicvideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=224,
            new_width=224,
            args=args)
        nb_classes = args.nb_classes

    elif args.data_set == 'dyadic_communication_mpigroup':
        mode = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        # debug 
        # print('====================================')
        # print('====================================')
        # print('====================================')
        # print(anno_path)
        # print('====================================')
        # print('====================================')
        # print('====================================')
        dataset = DyadicvideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=224,
            new_width=224,
            view_crop_mapping=mpigroup_cropping_map,
            corner_crop_size=None,
            limit_data=args.limit_data,
            args=args)
        nb_classes = args.nb_classes
    elif args.dataset == 'dyadic_communication_densepose':

        mode = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        # debug
        # print('====================================')
        # print('====================================')
        # print('====================================')
        # print(anno_path)
        # print('====================================')
        # print('====================================')
        # print('====================================')
        dataset = DensposeDyadicvideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=224,
            new_width=224,
            view_crop_mapping=mpigroup_cropping_map,
            corner_crop_size=None,
            limit_data=args.limit_data,
            densepose_extractor=densepose_extractor(args),
            args=args)
    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv') if anno_path is None else anno_path
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') if anno_path is None else anno_path
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') if anno_path is None else anno_path

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
        
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
