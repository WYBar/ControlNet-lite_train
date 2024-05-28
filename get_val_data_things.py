#
import os
import glob
import shutil
import numpy as np
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
import cv2
import json
from PIL import Image


# ROOT = "/data/WangYanbin"
ROOT = "../dataset"

# get names of all files in directory
list_of_files = glob.glob(os.path.join(ROOT, 'things_origin/*/*.jpg'))
r_indx = np.random.randint(0, len(list_of_files), 20000)
list_of_files = [list_of_files[i] for i in r_indx]
list_of_files = [file_path.replace('\\', '/') for file_path in list_of_files]
print(list_of_files)
# list_of_prompts = [list_of_prompts[i] for i in r_indx]

new_file_path = []
with open(os.path.join(ROOT, 'val_data.json'), 'wt') as f:
    for i, file_path in enumerate(list_of_files):
        # .replace('*', str(i), 1)
        # 拆分文件路径以获取目录和文件名
        # directory, old_filename = os.path.split(file_path)
        # 从文件名中提取文件的基本名称（不包含扩展名）
        # name, _ = os.path.splitext(old_filename)
        # 构建新的文件名，使用循环索引 i
        # new_filename = f"{i}.jpg"
        # 构建完整的新文件路径
        # new_file_path.append((os.path.join(ROOT, new_filename)).replace('\\', '/'))

        object_name = list_of_files[i].split('/')[-2]
        f.write(json.dumps({'target': f"images/{i}.jpg", 'source': f"edges/{i}.jpg", 'prompt': object_name, 'ds_label': 'things'}) + '\n')


os.makedirs(os.path.join(ROOT, 'edges/'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'images/'), exist_ok=True)
apply_canny = CannyDetector()

for i in range(len(list_of_files)):
    object_name = list_of_files[i].split('/')[-2]
    target = cv2.imread(list_of_files[i])
    target = resize_image(target, 256)
    detected_map = apply_canny(target, 100, 200)
    Image.fromarray(detected_map).save(os.path.join(ROOT, 'edges', f"{i}.jpg"))
    # detected_map = HWC3(detected_map)

    Image.fromarray(target).save(os.path.join(ROOT, 'images', f"{i}.jpg"))






