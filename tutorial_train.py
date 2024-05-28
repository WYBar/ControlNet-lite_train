from share import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tutorial_dataset import Dataset, ValDataset
from laion_dataset import LAIONDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import argparse
import os
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fill50k')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_time', type=str, default="01:00:00:00")
    parser.add_argument('--output_path', type=str, default='/output')
    parser.add_argument('--resume_path', type=str, default='/model/WangYanbin/control_sd15_lite/control_sd15_lite_ini.ckpt')
    parser.add_argument('--model_config', type=str, default='./models/cldm_lite_conv.yaml')
    parser.add_argument('--only_mid_control', type=bool, default=False)
    parser.add_argument('--sd_locked', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--logger_freq', type=int, default=500)
    parser.add_argument('--logger_dir', type=str, default='/output/wandb')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = vars(args)

    print(config)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config['model_config']).cpu()
    model.load_state_dict(load_state_dict(config['resume_path'], location='cpu'))
    model.learning_rate = config['learning_rate']
    model.sd_locked = config['sd_locked']
    model.only_mid_control = config['only_mid_control']

    # Misc
    # if config['dataset'] == 'fill50k':
    dataset = Dataset(config['dataset'])
    # else:
    #     dataset = LAIONDataset()

    if config['batch_size'] == 8:
        every_n_train_steps = 2500                  # 2500 * 8 = 20000
        batch_frequency = config['logger_freq'] * 4 # 500 * 4 * 8 = 16000
    else:
        every_n_train_steps = 5000                  # 5000 * 4 = 20000
        batch_frequency = config['logger_freq'] * 8 # 500 * 8 * 4 = 16000

    dataloader = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(ValDataset(config['dataset']), num_workers=0, batch_size=config['batch_size'], shuffle=False)
    logger = ImageLogger(save_dir=config['output_path'], batch_frequency=batch_frequency, val_dataloader=val_dataloader, log_images_kwargs = {"ddim_steps": 50, "N": config['batch_size']})
    # checkpoint = pl.callbacks.ModelCheckpoint(dirpath=config['output_path'], every_n_train_steps=every_n_train_steps, save_top_k=-1)
    # wandb_logger = WandbLogger(save_dir=config['output_path'], config=config, name="ControlNet_lite", project="ControlNet_lite", dir=config['logger_dir'])
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=32, callbacks=[logger],
        default_root_dir=config['output_path'], max_epochs=config['max_epochs'], max_time=config['max_time'])
    # logger=wandb_logger,
    # , checkpoint

    # Train!
    trainer.fit(model, dataloader)

    # Log final images
    # logger.log_img(model, None, 0)