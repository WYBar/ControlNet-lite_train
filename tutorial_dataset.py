import json # We need to use the JSON package to load the data, since the data is stored in JSON format
import cv2
import numpy as np

from torch.utils.data import Dataset

import os
from annotator.util import HWC3, resize_image, resize_image_square

class Dataset(Dataset):
    def __init__(self, dataset):
        if dataset == "fill50k":
            self.dataset_name = "fill50k";
        else:
            self.dataset_name = "things";

        self.data = []
        with open(os.path.join('/data/WangYanbin', self.dataset_name, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = cv2.imread(os.path.join('/data/WangYanbin', self.dataset_name, source_filename))
        target = cv2.imread(os.path.join('/data/WangYanbin', self.dataset_name, target_filename))

        source = resize_image(source, 256)
        target = resize_image(target, 256)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class ValDataset(Dataset):
    def __init__(self, dataset_name):
        self.data = []
        if dataset_name == 'fill50k':
            self.dataset_name = "fill50k_val";
            self.names = ['fill50k_val']
        else:
            # 'laion-art', 'CC3M'
            self.dataset_name = "things_val";
            self.names = ['things_val']
        for ds in self.names:
            with open(os.path.join('/data/WangYanbin', ds, 'prompt.json'), 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt'][:77]
        ds_label = self.dataset_name
        # ds_label = item['ds_label']

        source = cv2.imread(os.path.join('/data/WangYanbin', self.dataset_name, source_filename))
        target = cv2.imread(os.path.join('/data/WangYanbin', self.dataset_name, target_filename))

        # source = cv2.imread(source_filename)
        # target = cv2.imread(target_filename)

        source = resize_image_square(source, 256)
        target = resize_image_square(target, 256)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, ds_label=ds_label)


