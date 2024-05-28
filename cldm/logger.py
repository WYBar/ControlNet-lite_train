import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics.functional.multimodal import clip_score
from annotator.util import HWC3, resize_image
from aesthetics_predictor import AestheticsPredictor
from fid import fid_score
from functools import partial
import cv2
import time
import torch
import json
from torchvision.transforms import Normalize, Resize

ENABLE_PRINT = False

def print_if_enabled(*args, **kwargs):
    if ENABLE_PRINT:
        print(*args, **kwargs)

DS_LABEL = "fill50k"
def apply_canny(img):
    return cv2.Canny(img, 100, 200)

clip_score_fn = partial(clip_score, model_name_or_path="/model/WangYanbin/openai/clip-vit-base-patch16")

# target = torch.Tensor((batch['jpg'] + 1.0) * 127.5).to(torch.uint8)
# def calculate_clip_score(images, prompts):
#
#     clip_score = clip_score_fn(images, prompts).detach()
#     clip_score = np.array([np.round((clip_score.numpy()), 4)])
#     return clip_score
# if images.mode != 'RGB':
#     img = images.convert('RGB')
# normalize = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
# images = normalize(images)

def convert_to_python_types(value):
    if isinstance(value, np.ndarray):
        # 如果是 numpy 数组，先将其转换为 Python 列表
        value = value.tolist()

    if isinstance(value, list):
        # 如果是列表，递归处理每个元素
        return [convert_to_python_types(item) for item in value]
    elif isinstance(value, float):
        # 如果是 float32 类型，转换为 float
        return float(value)
    elif isinstance(value, int):
        # 如果是整数类型，转换为 int
        return int(value)
    else:
        # 其他情况，保持不变
        return value

def calculate_clip_score(images, prompts):
    images = images.float()
    # print(images.size()) # (batch, height, width, 3)
    print_if_enabled("Initial image shape:", images.shape) # (8, 3, 256, 256)
    print_if_enabled("Number of channels:", images.shape[1]) # 3
    # images = images.permute(0, 3, 1, 2)
    min_val = torch.min(images)
    max_val = torch.max(images)
    # 检查最小值和最大值是否分别为 0 和 255
    if min_val.item() != 0 or max_val.item() != 255:
        # 执行归一化操作
        images = (images - min_val) / (max_val - min_val)
        # 缩放至 [0, 255] 范围
        images = images * 255
    if images.shape[-2] != 256:
        print_if_enabled(images.size()) # (8, 256, 3, 256)
        images = Resize((256, 256))(images)
        print_if_enabled("Image shape after resizing:", images.shape) # ([8, 256, 256, 256])

    print("Image value range:", torch.min(images), "to", torch.max(images))
    images = images / 255.0
    print_if_enabled("Image value range after normalization:", torch.min(images), "to", torch.max(images))
    print_if_enabled("Image shape after normalization:", images.shape)
    clip_score = clip_score_fn(images, prompts).detach().cpu().numpy()
    clip_score = np.array([np.round(clip_score, 4)])
    return clip_score

def rmse(x, y):
    return np.mean(np.sqrt(np.mean((x - y) ** 2, axis=1)))

class ImageLogger(Callback):
    def __init__(self, save_dir, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, val_dataloader=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        self.save_dir = save_dir
        self.aesthetics_predictor = AestheticsPredictor()
        self.val_dataloader = val_dataloader
        if self.val_dataloader:
            self.get_target_clip_scores()

    def get_target_clip_scores(self):
        # 'things': [], 'laion-art': [], 'CC3M': []
        self.target_clip_scores = []
        for batch in self.val_dataloader:
            target = torch.Tensor((batch['jpg'] + 1.0) * 127.5).to(torch.uint8)
            self.target_clip_scores.append(calculate_clip_score(target, batch['txt']))

        print_if_enabled(self.target_clip_scores)

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.save_dir, "image_log", split)
        ds_label = DS_LABEL
        del images['ds_label']
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images_to_log = images[k][:N]
            grid = torchvision.utils.make_grid(images_to_log, nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = grid.astype(np.uint8)
            filename = "{}_{}_gs-{:06}_e-{:06}_b-{:06}.png".format(DS_LABEL, k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_metrics_local(self, metrics, global_step, current_epoch):
        # 定义保存文件的目录
        save_dir = os.path.join(self.save_dir, "metrics_log")
        os.makedirs(save_dir, exist_ok=True)
        # 构建文件名
        filename = "metrics_gs-{:06}_e-{:06}.json".format(global_step, current_epoch)
        file_path = os.path.join(save_dir, filename)

        # 将 metrics 字典中的 float32 数据转换为 float 或 int
        for key, value in metrics.items():
            if isinstance(value, (float, int, str, bool, type(None))):
                continue  # 已经是基本数据类型，无需转换
            elif isinstance(value, (list, tuple, set)):
                # 递归转换列表、元组、集合中的每个元素
                metrics[key] = [convert_to_python_types(item) for item in value]
            elif isinstance(value, dict):
                # 递归转换字典中的每个值
                metrics[key] = {k: convert_to_python_types(v) for k, v in value.items()}
            else:
                # 尝试将其他类型转换为字符串
                metrics[key] = str(value)

        # 将 metrics 字典转换为 JSON 格式并保存
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            # logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            all_images = []
            some_images = []
            sampling_times = []
            with torch.no_grad():
                if self.val_dataloader:
                    for i, dl_batch in enumerate(self.val_dataloader):
                        s = time.time()
                        print_if_enabled(f"all_images epoch: {i}")
                        imgs = pl_module.log_images(dl_batch, split=split, **self.log_images_kwargs)
                        sampling_times.append(time.time() - s)
                        imgs['ds_label'] = DS_LABEL
                        # imgs['target'] = dl_batch['jpg']
                        all_images.append(imgs)
                # else:

                s = time.time()
                imgs = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                sampling_times.append(time.time() - s)
                imgs['ds_label'] = DS_LABEL
                # imgs['target'] = batch['jpg']
                some_images.append(imgs)

            for i, images in enumerate(some_images):
                print_if_enabled("Image value range clip epoch:", torch.min(some_images[i]['samples_cfg_scale_9.00']),
                                 "to",
                                 torch.max(some_images[i]['samples_cfg_scale_9.00']))
                print_if_enabled("log Image value:", some_images)
                for k in images:
                    if k != 'ds_label':
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)
                            if self.rescale:
                                images[k] = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                            images[k] = (images[k] * 255).to(torch.uint8)

                self.log_local(split, images,
                               pl_module.global_step, pl_module.current_epoch, i)

            # log metrics
            metrics = {"clip_score": [], 'delta_clip_score': [], 'aesthetics_score': [], 'sampling_time': np.round(np.mean(sampling_times), 2), 'fid_score': []}
            if self.val_dataloader:
                for i, dl_batch in enumerate(self.val_dataloader):
                    # all_images[i]['samples_cfg_scale_9.00'] returns torch tensor of
                    # shape batch, 3, width, height with pixel range [0, 255] (uint8)
                    # CLIP expects RGB batch of shape batch, 3, 256, 256 with pixel range [0, 1] normalized with
                    # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                    # https://github.com/openai/CLIP/blob/main/clip/clip.py
                    print_if_enabled(f"clip_score epoch: {i}")
                    print_if_enabled("Initial image shape:", all_images[i]['samples_cfg_scale_9.00'].shape)
                    print_if_enabled("Image value range clip epoch:", torch.min(all_images[i]['samples_cfg_scale_9.00']), "to",
                          torch.max(all_images[i]['samples_cfg_scale_9.00']))
                    print_if_enabled("log Image value:", all_images[i])
                    metrics['aesthetics_score'].append(np.mean(self.aesthetics_predictor.inference(all_images[i]['samples_cfg_scale_9.00'].float() / 255.0).cpu().detach().numpy()))
                    # try:                                                       ['samples_cfg_scale_9.00']
                    metrics['clip_score'].append(calculate_clip_score(all_images[i]['samples_cfg_scale_9.00'], dl_batch['txt']))
                    # except:
                    #     metrics['clip_score'].append(0)

                # try:
                metrics['delta_clip_score'] = np.mean([metrics['clip_score'][i] - self.target_clip_scores[i] for i in range(len(metrics['clip_score']))])
                metrics['clip_score'] = np.mean(metrics['clip_score'])
                # except:
                #     pass

                metrics['aesthetics_score'] = np.mean(metrics['aesthetics_score'])
                try:  # 有时会报错 "sqrtm: array must not contain infs or NaNs"
                    metrics['fid_score'] = fid_score.calculate_fid(all_images, self.val_dataloader, batch_size=32, 
                        device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048, num_workers=4)
                except:
                    pass

                # self.log_local(split, images, pl_module.global_step, pl_module.current_epoch, i)
                self.log_metrics_local(metrics, pl_module.global_step, pl_module.current_epoch)
                # pl_module.logger.log_metrics(metrics, step=pl_module.global_step)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_train_end(self, trainer, pl_module):
        print('ON TRAIN END')
        self.log_img(pl_module, None, 0, split="train")

    def on_exception(self, trainer, pl_module, exception):
        print('ON EXCEPTION')
        self.log_img(pl_module, None, 0, split="train")
