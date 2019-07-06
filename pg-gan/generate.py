import tqdm
import traceback
import shutil
import argparse
import os
import copy
import matplotlib
from torch_utils import generate_run_base_dir
import torchvision

# matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from mnist_example import hypersphere

from torch import nn
import torch
from torch.nn import functional as F
from torch_utils import make_multi_gpu
from torch_utils import save_args
from torch_utils import set_seed
from torch_utils import set_cuda_device
from torch_utils import ensure_dir
from torchvision import datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

from model import Generator, Discriminator
from progressBar import printProgressBar

from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outd", default="Results", help="directory to save results")
    parser.add_argument(
        "--outf", default="Images", help="folder to save synthetic images"
    )
    parser.add_argument(
        "--nch", type=int, default=4, help="base number of channel for networks"
    )
    parser.add_argument(
        "--cuda-device-id",
        nargs="+",
        type=int,
        default=[0],
        help="Cuda device ids. E.g. [0,1,2]. Use -1 for all GPUs available.",
    )
    parser.add_argument("--BN", action="store_true", help="use BatchNorm in G and D")
    parser.add_argument("--WS", action="store_true", help="use WeightScale in G and D")
    parser.add_argument("--PN", action="store_true", help="use PixelNorm in G")
    parser.add_argument(
        "--lambdaGP", type=float, default=10, help="lambda for gradient penalty"
    )
    parser.add_argument(
        "--gamma", type=float, default=1, help="gamma for gradient penalty"
    )
    parser.add_argument(
        "--e_drift",
        type=float,
        default=0.001,
        help="epsilon drift for discriminator loss",
    )
    parser.add_argument(
        "--aux-lambda",
        type=float,
        default=1,
        help="Auxillary classifier loss lambda value",
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=4,
        help="Max resolution. 0->4x4, 1->8x8, 3->32x32, 5->128x128 ...",
    )
    parser.add_argument(
        "--savenum", type=int, default=16, help="number of examples images to save"
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Tag to identify runs in the result directory and tensorboard overviews",
        required=True,
    )
    parser.add_argument(
        "--model-path", type=str, help="Generator model path.", required=True
    )

    return parser.parse_args()


def generate_images(n, tag, label):
    for i in tqdm.tqdm(range(n)):
        fake_images = None
        z_save = hypersphere(
            num_classes=2,
            batch_size=savenum,
            size=opt.nch * 32,
            device=DEVICE,
            # label=torch.arange(savenum) % 2,
            label=label,
        )
        fake_images = G(z_save)

        fake_images = fake_images * data_stds.view(1, 3, 1, 1) + data_means.view(
            1, 3, 1, 1
        )
        save_image(
            fake_images,
            os.path.join("test-gen", f"fake-{tag}-{i}.png"),
            nrow=8,
            pad_value=0,
            normalize=True,
            range=(0, 1),
            scale_each=True,
        )


def save(fake_images, path):
    fake_images = fake_images * data_stds.view(1, 3, 1, 1) + data_means.view(1, 3, 1, 1)
    save_image(
        fake_images,
        path,
        nrow=8,
        pad_value=0,
        normalize=True,
        range=(0, 1),
        scale_each=True,
    )


def interp_circular(v1, v2, alpha):
    """
    Interpolate between two noise vectors on the hyper-circle.

    Args:
        v1: Start noise vector.
        v2: End noise vector.
        alpha: Linear interpolation factor.

    Returns:
        np.ndarray: alpha * v1 + (1 - alpha) * v2 / (alpha * ||v1|| + (1 - alpha) * ||v2||)
    """
    v = alpha * v1 + (1 - alpha) * v2
    norm = alpha * torch.norm(v1, dim=1) + (1 - alpha) * torch.norm(v2, dim=1)
    return v / norm.view(-1, 1, 1, 1) * np.sqrt(2)


def gen_from_seed(seed: int, cls):
    set_seed(seed)
    if cls == "pizza":
        label = torch.zeros(0)
    elif cls == "burger":
        label = torch.ones(1)

    z_save = hypersphere(
        num_classes=2, batch_size=1, size=opt.nch * 32, device=DEVICE, label=label
    )
    return G(z_save)


def interpolate_between_seeds(seed_start, label_start, seed_end, label_end):
    basedir = os.path.join(
        "interps", f"seed-pizza={seed_start}", f"seed-burger={seed_end}"
    )
    ensure_dir(basedir)
    # Generate samples with class=0
    noise_start_label = torch.ones(1) * label_start
    set_seed(seed_start)
    noise_start = hypersphere(
        num_classes=2,
        batch_size=1,
        size=opt.nch * 32,
        device=DEVICE,
        label=noise_start_label,
    )

    # Generate samples with class=1
    noise_end_label = torch.ones(1) * label_end
    set_seed(seed_end)
    noise_end = hypersphere(
        num_classes=2,
        batch_size=1,
        size=opt.nch * 32,
        device=DEVICE,
        label=noise_end_label,
    )

    # Alpha from 0 to 1 in 300 steps
    for idx, alpha in tqdm.tqdm(enumerate(np.linspace(0, 1, num=300, endpoint=True))):
        noise = interp_circular(noise_start, noise_end, alpha).to(DEVICE)
        fake = G(noise)
        save(fake, path=os.path.join(basedir, f"interp-{idx:>03}.png"))


def gen_multiple_seeds(n):
    basedir = "gens"
    pizzadir = os.path.join(basedir, "pizza")
    burgerdir = os.path.join(basedir, "burger")
    ensure_dir(pizzadir)
    ensure_dir(burgerdir)

    for i in tqdm.tqdm(range(n)):
        fake_pizza = gen_from_seed(i, "pizza")
        save(fake_pizza, path=os.path.join(pizzadir, f"fake-seed={i:>03}.png"))

        fake_burger = gen_from_seed(i, "burger")
        save(fake_burger, path=os.path.join(burgerdir, f"fake-seed={i:>03}.png"))


def make_interpolation(G, bs):
    """
    Generate interpolations.

    Args:
        netG: Generator network.
    """
    # Generate samples with class=0
    noise_start_label = torch.zeros(bs)
    noise_start = hypersphere(
        num_classes=2,
        batch_size=bs,
        size=opt.nch * 32,
        device=DEVICE,
        label=noise_start_label,
    )

    # Generate samples with class=1
    noise_end_label = torch.ones(bs)
    noise_end = hypersphere(
        num_classes=2,
        batch_size=bs,
        size=opt.nch * 32,
        device=DEVICE,
        label=noise_end_label,
    )

    # Alpha from 0 to 1 in 100 steps
    for idx, alpha in tqdm.tqdm(enumerate(np.linspace(0, 1, num=20, endpoint=True))):
        noise = interp_circular(noise_start, noise_end, alpha).to(DEVICE)
        fake = G(noise)
        print(f"Fake shape at idx={idx:>03}, alpha={alpha:.3f}")
        print(fake.shape)
        for i in range(bs):
            ensure_dir(os.path.join("test-gen", str(i)))
            save(
                fake[[i], :, :, :],
                path=os.path.join("test-gen", str(i), f"fake-{idx:>03}.png"),
            )


if __name__ == "__main__":
    opt = parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        G = torch.load(opt.model_path).to(DEVICE)
    else:
        G = torch.load(opt.model_path, map_location="cpu").to(DEVICE)
    G.eval()

    savenum = 64

    data_means = torch.tensor([2.0581, 2.0569, 2.2445]).to(DEVICE)
    data_stds = torch.tensor([1.0557, 1.1210, 1.2443]).to(DEVICE)

    with torch.no_grad():
        # generate_images(1, "burger", label=torch.ones(savenum))
        # generate_images(1, "pizza", label=torch.zeros(savenum))

        # make_interpolation(G, 32)
        # gen_multiple_seeds(1000)
        good_pizzas = [
            (502, 0),
            (158, 0),
            (341, 0),
            (574, 0),
            (998, 0),
            (9, 1),
            (97, 1),
            (112, 1),
            (141, 1),
            (184, 1),
            (239, 1),
            (282, 1),
            (315, 1),
            (356, 1),
            (411, 1),
            (545, 1),
            (597, 1),
            (663, 1),
            (712, 1),
            (742, 1),
            (961, 1),
        ]
        good_burgers = [
            (14, 1),
            (16, 1),
            (48, 1),
            (60, 1),
            (103, 1),
            (915, 1),
            (912, 1),
            (802, 1),
            (768, 1),
            (740, 1),
            (579, 1),
        ]
        for seed_start, label_start in tqdm.tqdm(good_pizzas):
            for seed_end, label_end in good_burgers:
                interpolate_between_seeds(seed_start, label_start, seed_end, label_end)
