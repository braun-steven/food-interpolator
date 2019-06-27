import argparse
import os
import copy
import matplotlib
from torch_utils import generate_run_base_dir

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F
from torch_utils import make_multi_gpu
from torch_utils import save_args
from torchvision import datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

from model import Generator, Discriminator
from progressBar import printProgressBar
from utils import weights_init, GradientPenalty, Progress, exp_mov_avg, hypersphere

from time import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="DATA", help="directory containing the data"
)
parser.add_argument("--outd", default="Results", help="directory to save results")
parser.add_argument("--outf", default="Images", help="folder to save synthetic images")
parser.add_argument("--outl", default="Losses", help="folder to save Losses")
parser.add_argument("--outm", default="Models", help="folder to save models")

parser.add_argument(
    "--workers", type=int, default=8, help="number of data loading workers"
)
parser.add_argument(
    "--batchSizes",
    nargs="+",
    type=int,
    default=[16, 16, 16, 16],
    help="list of batch sizes during the training",
)
parser.add_argument(
    "--nch", type=int, default=4, help="base number of channel for networks"
)
parser.add_argument("--BN", action="store_true", help="use BatchNorm in G and D")
parser.add_argument("--WS", action="store_true", help="use WeightScale in G and D")
parser.add_argument("--PN", action="store_true", help="use PixelNorm in G")

parser.add_argument(
    "--n-iter",
    type=int,
    default=1,
    help="number of epochs to train before changing the progress",
)
parser.add_argument(
    "--lambdaGP", type=float, default=10, help="lambda for gradient penalty"
)
parser.add_argument("--gamma", type=float, default=1, help="gamma for gradient penalty")
parser.add_argument(
    "--e_drift", type=float, default=0.001, help="epsilon drift for discriminator loss"
)
parser.add_argument(
    "--saveimages",
    type=int,
    default=1,
    help="number of epochs between saving image examples",
)
parser.add_argument(
    "--max-res",
    type=int,
    default=4,
    help="Max resolution. 0->4x4, 1->8x8, 3->32x32, 5->128x128 ...",
)
parser.add_argument(
    "--savenum", type=int, default=64, help="number of examples images to save"
)
parser.add_argument(
    "--savemodel", type=int, default=10, help="number of epochs between saving models"
)
parser.add_argument(
    "--savemaxsize",
    action="store_true",
    help="save sample images at max resolution instead of real resolution",
)
parser.add_argument(
    "--tag",
    default="",
    type=str,
    help="Tag to identify runs in the result directory and tensorboard overviews",
    required=True,
)

opt = parser.parse_args()
opt.outd = generate_run_base_dir(opt.outd, time(), tag=opt.tag)
print(opt)
save_args(opt, opt.outd)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_RES = opt.max_res  # for 32x32 output

data_means = torch.tensor([0.4494, 0.4235, 0.4196]).to(DEVICE)
data_stds = torch.tensor([0.9628, 0.9711, 1.0579]).to(DEVICE)


def make_dataset(path_to_folder, classes):
    d = 4 * 2 ** opt.max_res
    # Data transformations
    data_transform = transforms.Compose(
        [
            transforms.Resize((d, d)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_means.cpu().numpy(), std=data_stds.cpu().numpy()
            ),
        ]
    )

    classes = set(classes)

    # Define dataset via imagefolder
    dataset = datasets.ImageFolder(
        root=path_to_folder, transform=data_transform  # , is_valid_file=filter_data
    )
    ####### {{{
    # Custom code copied from datasets.folder.DatasetFolder to support partial selection of classes
    TRAINING_CLASS = ["pizza", "burger"]
    VAL_CLASS = TRAINING_CLASS
    NUM_CLASSES = len(VAL_CLASS)
    classes = TRAINING_CLASS
    class_to_idx = {TRAINING_CLASS[i]: i for i in range(NUM_CLASSES)}

    samples = datasets.folder.make_dataset(
        dataset.root, class_to_idx, extensions=["jpg", "jpeg", "png"]  # , filter_data
    )
    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + dataset.root + "\n"))

    dataset.loader = datasets.folder.default_loader
    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.samples = samples
    dataset.targets = [s[1] for s in samples]
    ####### }}}
    return dataset


# Get dataset
transform = transforms.Compose(
    [
        # resize to 32x32
        transforms.Pad((2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# dataset = MNIST(opt.data, download=True, train=True, transform=transform)
dataset = make_dataset(opt.data, ["pizza", "burger"])

# creating output folders
if not os.path.exists(opt.outd):
    os.makedirs(opt.outd)
for f in [opt.outf, opt.outl, opt.outm]:
    if not os.path.exists(os.path.join(opt.outd, f)):
        os.makedirs(os.path.join(opt.outd, f))

# Model creation and init
n_channels = 3
G = Generator(
    max_res=MAX_RES, nch=opt.nch, nc=n_channels, bn=opt.BN, ws=opt.WS, pn=opt.PN
).to(DEVICE)
D = Discriminator(max_res=MAX_RES, nch=opt.nch, nc=n_channels, bn=opt.BN, ws=opt.WS).to(
    DEVICE
)
devices = list(range(torch.cuda.device_count()))
if not opt.WS:
    # weights are initialized by WScale layers to normal if WS is used
    G.apply(weights_init)
    D.apply(weights_init)
G = make_multi_gpu(G, devices)
D = make_multi_gpu(D, devices)
Gs = copy.deepcopy(G)

optimizerG = Adam(G.parameters(), lr=1e-3, betas=(0, 0.99))
optimizerD = Adam(D.parameters(), lr=1e-3, betas=(0, 0.99))

GP = GradientPenalty(opt.batchSizes[0], opt.lambdaGP, opt.gamma, device=DEVICE)

epoch = 0
global_step = 0
total = 2
d_losses = np.array([])
d_losses_W = np.array([])
g_losses = np.array([])
P = Progress(opt.n_iter, MAX_RES, opt.batchSizes)

z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))

P.progress(epoch, 1, total)
GP.batchSize = P.batchSize


# Creation of DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=P.batchSize,
    shuffle=True,
    num_workers=opt.workers,
    drop_last=True,
    pin_memory=True,
)

while True:
    t0 = time()

    lossEpochG = []
    lossEpochD = []
    lossEpochD_W = []

    G.train()
    cudnn.benchmark = True

    P.progress(epoch, 1, total)

    if P.batchSize != data_loader.batch_size:
        # update batch-size in gradient penalty
        GP.batchSize = P.batchSize
        # modify DataLoader at each change in resolution to vary the batch-size as the resolution increases
        data_loader = DataLoader(
            dataset,
            batch_size=P.batchSize,
            shuffle=True,
            num_workers=opt.workers,
            drop_last=True,
            pin_memory=True,
        )

    total = len(data_loader)

    for i, (images, _) in enumerate(data_loader):
        P.progress(epoch, i + 1, total + 1)
        global_step += 1

        # Build mini-batch
        images = images.to(DEVICE)
        images = P.resize(images)

        # ============= Train the discriminator =============#

        # zeroing gradients in D
        D.zero_grad()
        # compute fake images with G
        z = hypersphere(torch.randn(P.batchSize, opt.nch * 32, 1, 1, device=DEVICE))
        with torch.no_grad():
            fake_images = G(z, P.p)

        # compute scores for real images
        D_real = D(images, P.p)
        D_realm = D_real.mean()

        # compute scores for fake images
        D_fake = D(fake_images, P.p)
        D_fakem = D_fake.mean()

        # compute gradient penalty for WGAN-GP as defined in the article
        gradient_penalty = GP(D, images.data, fake_images.data, P.p)

        # prevent D_real from drifting too much from 0
        drift = (D_real ** 2).mean() * opt.e_drift

        # Backprop + Optimize
        d_loss = D_fakem - D_realm
        d_loss_W = d_loss + gradient_penalty + drift
        d_loss_W.backward()
        optimizerD.step()

        lossEpochD.append(d_loss.item())
        lossEpochD_W.append(d_loss_W.item())

        # =============== Train the generator ===============#

        G.zero_grad()

        z = hypersphere(torch.randn(P.batchSize, opt.nch * 32, 1, 1, device=DEVICE))
        fake_images = G(z, P.p)
        # compute scores with new fake images
        G_fake = D(fake_images, P.p)
        G_fakem = G_fake.mean()
        # no need to compute D_real as it does not affect G
        g_loss = -G_fakem

        # Optimize
        g_loss.backward()
        optimizerG.step()

        lossEpochG.append(g_loss.item())

        # update Gs with exponential moving average
        exp_mov_avg(Gs, G, alpha=0.999, global_step=global_step)

        printProgressBar(
            i + 1,
            total + 1,
            length=20,
            prefix=f"Epoch {epoch} ",
            suffix=f", d_loss: {d_loss.item():.3f}"
            f", d_loss_W: {d_loss_W.item():.3f}"
            f", GP: {gradient_penalty.item():.3f}"
            f", progress: {P.p:.2f}",
        )

    printProgressBar(
        total,
        total,
        done=f"Epoch [{epoch:>3d}]  d_loss: {np.mean(lossEpochD):.4f}"
        f", d_loss_W: {np.mean(lossEpochD_W):.3f}"
        f", progress: {P.p:.2f}, time: {time() - t0:.2f}s",
    )

    d_losses = np.append(d_losses, lossEpochD)
    d_losses_W = np.append(d_losses_W, lossEpochD_W)
    g_losses = np.append(g_losses, lossEpochG)

    np.save(os.path.join(opt.outd, opt.outl, "d_losses.npy"), d_losses)
    np.save(os.path.join(opt.outd, opt.outl, "d_losses_W.npy"), d_losses_W)
    np.save(os.path.join(opt.outd, opt.outl, "g_losses.npy"), g_losses)

    cudnn.benchmark = False
    if not (epoch + 1) % opt.saveimages:
        # plotting loss values, g_losses is not plotted as it does not represent anything in the WGAN-GP
        ax = plt.subplot()
        ax.plot(
            np.linspace(0, epoch + 1, len(d_losses)),
            d_losses,
            "-b",
            label="d_loss",
            linewidth=0.1,
        )
        ax.legend(loc=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Progress: {P.p:.2f}")
        plt.savefig(
            os.path.join(opt.outd, opt.outl, f"Epoch_{epoch}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.clf()

        # Save sampled images with Gs
        Gs.eval()
        # z = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))

        with torch.no_grad():
            fake_images = Gs(z_save, P.p)
            if opt.savemaxsize:
                if fake_images.size(-1) != 4 * 2 ** MAX_RES:
                    fake_images = F.upsample(fake_images, 4 * 2 ** MAX_RES)
        save_image(
            fake_images,
            os.path.join(opt.outd, opt.outf, f"fake_images-{epoch:04d}-p{P.p:.2f}.png"),
            nrow=8,
            pad_value=0,
            normalize=True,
            range=(-1, 1),
        )

    if P.p >= P.pmax and not epoch % opt.savemodel:
        torch.save(
            G, os.path.join(opt.outd, opt.outm, f"G_nch-{opt.nch}_epoch-{epoch}.pth")
        )
        torch.save(
            D, os.path.join(opt.outd, opt.outm, f"D_nch-{opt.nch}_epoch-{epoch}.pth")
        )
        torch.save(
            Gs, os.path.join(opt.outd, opt.outm, f"Gs_nch-{opt.nch}_epoch-{epoch}.pth")
        )

    epoch += 1
