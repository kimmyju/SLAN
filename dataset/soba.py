import torch
import torch.nn
import logging
import numpy as np

from PIL import Image
from pathlib import Path
from os.path import splitext, isfile, join
from os import listdir
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class SOBA(torch.utils.data.Dataset) :
    
    def __init__(
            self,
            sample_path, 
            obj_label_path,
            sha_label_path,
            light_path, 
            scale : float = 0.5,
            mask_suffix = "_mask"
    ) :
        self.sample_path = Path(sample_path)
        self.obj_label_path = Path(obj_label_path)
        self.sha_label_path = Path(sha_label_path)
        self.light_path = Path(light_path)
        self.scale = scale 
        self.mask_suffix = mask_suffix
        self.ids = list()

        for file in listdir(self.sample_path) :
            if (isfile(join(self.sample_path, file))) :
                self.ids.append(splitext(file)[0])
        self.mask_values = [0, 1]

        self.light_ids = {}

        with open(self.light_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                if len(data) == 2:
                    file_name, light_number = data
                    self.light_ids[file_name] = int(light_number)

    def __len__(self) :
        return len(self.ids)

    def __getitem__(self, idx) :
        file_name = self.ids[idx]
        file_expansion = ".*"
        sample_file = list(self.sample_path.glob(file_name + file_expansion))
        obj_label_file = list(self.obj_label_path.glob(file_name + self.mask_suffix + file_expansion))
        sha_label_file = list(self.sha_label_path.glob(file_name + self.mask_suffix + file_expansion))

        sample = load_image(sample_file[0])
        obj_mask = load_image(obj_label_file[0])
        sha_mask = load_image(sha_label_file[0])
        
        sample = self.preprocess(self.mask_values, sample, self.scale, is_mask=  False)
        obj_mask = self.preprocess(self.mask_values, obj_mask, self.scale, is_mask = True)
        sha_mask = self.preprocess(self.mask_values, sha_mask, self.scale, is_mask = True)
        return {
            "image" : torch.as_tensor(sample.copy()).float().contiguous(),
            "obj_mask" : torch.as_tensor(obj_mask.copy()).long().contiguous(),
            "sha_mask" : torch.as_tensor(sha_mask.copy()).long().contiguous(),
            "light" : self.light_ids[file_name],
        }

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img