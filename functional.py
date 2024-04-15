import numbers
from PIL import Image
import numpy as np
import numpy as np
import PIL
from PIL import Image
import torch


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped

def pad_clip(clip, pad_top, pad_bottom, pad_left, pad_right, pad_value):
    if isinstance(clip[0], np.ndarray):
        padded = [
            np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                   mode='constant', constant_values=pad_value) for img in clip
        ]
        
    elif isinstance(clip[0], PIL.Image.Image):
        padded = [
            Image.fromarray(np.pad(np.array(img), ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                  mode='constant', constant_values=pad_value))
            for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return padded





def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            pil_inter = Image.BILINEAR
        else:
            pil_inter = Image.NEAREST
        scaled = [
            np.array(Image.fromarray(img).resize(size, resample=pil_inter)) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

def load_image_from_file(file_path):
    img = Image.open(file_path)
    return np.array(img)

def test_crop():
    # Replace 'your_local_image_path' with the path to your local image file
    local_image_path = 'figs/videomae.jpg'
    image = load_image_from_file(local_image_path)

    # Example usage of resize_clip function
    resized_image = resize_clip([image], size=100, interpolation='bilinear')[0]

    # Display the original and resized images
    Image.fromarray(image).show(title='Original Image')
    Image.fromarray(resized_image).show(title='Resized Image')


if __name__== "__main__":
    test_crop()
