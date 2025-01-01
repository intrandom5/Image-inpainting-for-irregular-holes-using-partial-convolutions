from model import UNet
from inpainting import inpainting_train
from dataset import get_loader

from types import SimpleNamespace
import argparse
import torch
import yaml
import glob
import os


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    set_seed(58)
    train_imgs = os.path.join(config.train_dir, "*.png")
    train_imgs = glob.glob(train_imgs)
    valid_imgs = os.path.join(config.valid_dir, "*.png")
    valid_imgs = glob.glob(valid_imgs)

    # load config.models
    generator = UNet(in_channels=3, out_channels=3, start_dim=64)

    train_loader = get_loader(train_imgs, batch_size=config.batch_size, shuffle=True)
    valid_loader = get_loader(valid_imgs, batch_size=config.batch_size, shuffle=False)

    # Train start
    print("train started")
    inpainting_train(train_loader, valid_loader, generator, config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="config.yaml", help="config file path(.yaml)")
    args = parser.parse_args()
    with open(args.conf, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config = SimpleNamespace(**config_dict)

    main(config)