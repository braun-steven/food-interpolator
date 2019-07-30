import random
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


def save(fake_images, path, nrow=8):
    fake_images = fake_images * data_stds.view(1, 3, 1, 1) + data_means.view(1, 3, 1, 1)
    save_image(
        fake_images,
        path,
        nrow=nrow,
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
        np.ndarray: alpha * v2 + (1 - alpha) * v1 / (alpha * ||v2|| + (1 - alpha) * ||v1||)
    """
    v = alpha * v2 + (1 - alpha) * v1
    norm = alpha * torch.norm(v2, dim=1) + (1 - alpha) * torch.norm(v1, dim=1)
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


def interpolate_between_seeds_iter(seed_start, seed_end):
    basedir = os.path.join(
        "interps", f"seed-pizza={seed_start}", f"seed-burger={seed_end}"
    )
    ensure_dir(basedir)
    # Generate samples with class=0
    noise_start_label = torch.ones(1)

    noise_start = []
    for seed in seed_start:
        set_seed(seed)
        noise_start.append(
            hypersphere(
                num_classes=2,
                batch_size=1,
                size=opt.nch * 32,
                device=DEVICE,
                label=noise_start_label,
            )
        )
    noise_start = torch.cat(noise_start, dim=0)

    # Generate samples with class=1
    noise_end_label = torch.ones(1)
    noise_end = []
    for seed in seed_end:
        set_seed(seed)
        noise_end.append(
            hypersphere(
                num_classes=2,
                batch_size=1,
                size=opt.nch * 32,
                device=DEVICE,
                label=noise_end_label,
            )
        )
    noise_end = torch.cat(noise_end, dim=0)

    # Alpha from 0 to 1 in 300 steps
    for idx, alpha in enumerate(np.linspace(0, 1, num=150, endpoint=True)):
        noise = interp_circular(noise_start, noise_end, alpha).to(DEVICE)
        fake = G(noise)
        yield fake


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

    data_means = torch.tensor([0.6250, 0.5109, 0.4063]).to(DEVICE)
    data_stds = torch.tensor([0.2323, 0.2430, 0.2522]).to(DEVICE)

    ensure_dir("interpolations")
    set_seed(0)
    with torch.no_grad():
        # generate_images(1, "burger", label=torch.ones(savenum))
        # generate_images(1, "pizza", label=torch.zeros(savenum))

        # make_interpolation(G, 32)
        # gen_multiple_seeds(1000)
        good_pizzas = [
            0,
            9,
            33,
            47,
            997,
            88,
            87,
            152,
            174,
            209,
            222,
            272,
            276,
            367,
            397,
            442,
            537,
            565,
            597,
            623,
            665,
            693,
            811,
            973,
            997,
        ]
        good_burgers = [
            54,
            66,
            80,
            94,
            194,
            217,
            251,
            255,
            266,
            277,
            310,
            325,
            349,
            353,
            404,
            446,
            474,
            497,
            516,
            520,
            568,
            612,
            626,
            650,
            669,
            677,
            840,
            857,
            885,
            918,
        ]
        rand = random.Random(0)

        def get_random(size, a, b):
            rand.shuffle(a)
            arand = a[:size]
            rand.shuffle(b)
            brand = b[:size]
            return arand, brand

        def interp(pizza_seeds, burger_seed, running_counter, d):
            # Get interpolations
            for fake_interp in interpolate_between_seeds_iter(pizza_seeds, burger_seed):
                path = os.path.join(d, f"fake-{running_counter:>04}.png")
                save(fake_interp, path, nrow=2)
                running_counter += 1
            return running_counter

        def make_interpolations(outpath, a, b):

            d = outpath
            ensure_dir(d)
            N = 4
            initial_pizza, initial_burger = get_random(N, a, b)
            interp_count = 0
            interp_count = interp(initial_pizza, initial_burger, interp_count, d)
            prev_seeds = initial_burger
            # Make dir
            for i in tqdm.trange(20):
                rand_pizza_seeds, rand_burger_seeds = get_random(N, a, b)
                print(f"Seed 1: {rand_pizza_seeds}")
                print(f"Seed 2: {rand_burger_seeds}")
                interp_count = interp(prev_seeds, rand_pizza_seeds, interp_count, d)
                interp_count = interp(
                    rand_pizza_seeds, rand_burger_seeds, interp_count, d
                )
                prev_seeds = rand_burger_seeds

            interp_count = interp(prev_seeds, initial_pizza, interp_count, d)

        print("Starting pizza to burger interpolations")
        make_interpolations(
            "interpolations/pizza-burger-cycle", good_pizzas, good_burgers
        )

        # print("Starting pizza to pizza interpolations")
        # make_interpolations(
        #     "interpolations/pizza-pizza-cycle", good_pizzas, good_pizzas
        # )
        # print("Starting burger to burger interpolations")
        # make_interpolations(
        #     "interpolations/burger-burger-cycle", good_burgers, good_burgers
        # )
        # print("Starting totally random interpolations")
        # make_interpolations(
        #     "interpolations/random", list(range(100000)), list(range(100000))
        # )

        # for seed_start in tqdm.tqdm(good_pizzas):
        #     for seed_end in good_burgers:
        #         interpolate_between_seeds_iter(seed_start, seed_end)
