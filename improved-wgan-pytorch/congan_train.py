from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import datetime
import functools
import os
import sys
from time import time

import gpustat
import numpy as np
import torch
import torch.nn.init as init
import torchvision
from tensorboardX import SummaryWriter
from torch import autograd, nn, optim
from torch.autograd import grad
from torchvision import datasets, transforms

import libs as lib
import libs.plot
from models.conwgan import GoodDiscriminator, GoodGenerator, MyConvo2d, set_dims
import logging

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def setup_logging(filename: str = "log.txt", level: str = "INFO"):
    """
    Setup global loggers.

    Args:
        filename: Log file destination.
        level: Log level.
    """
    # Make sure the directory actually exists
    ensure_dir(os.path.dirname(filename))

    # Check if previous log exists since logging.FileHandler only appends
    if os.path.exists(filename):
        os.remove(filename)

    logging.basicConfig(
        level=logging.getLevelName(level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=filename),
        ],
    )


def set_cuda_device(cuda_device_id):
    """
    Set the visible cuda devices.

    Args:
        cuda_device_id (List[int]): Cuda device ids.

    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cuda_device_id])


def make_multi_gpu(model):
    num_cuda_devices = torch.cuda.device_count()
    multi_gpu = num_cuda_devices > 1

    # Check if multiple cuda devices are selected
    if multi_gpu:
        logger.info("Using multiple gpus")

        # Select all devices
        cuda_device_id = list(range(num_cuda_devices))

        # Check if multiple cuda devices are available
        if num_cuda_devices > 1:
            logger.info("Running experiment on the following GPUs: %s" % cuda_device_id)

            # Transform model into data parallel model on all selected cuda deviecs
            model = torch.nn.DataParallel(model, device_ids=cuda_device_id)
        else:
            logger.info(
                "Attempted to run the experiment on multiple GPUs while only %s GPU was available"
                % num_cuda_devices
            )
    return model


def time_delta_now(ts: float) -> str:
    """
    Convert a timestamp into a human readable timestring (%H:%M:%S).
    Args:
        ts (float): Timestamp.

    Returns:
        Human readable timestring.
    """
    a = ts
    b = time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"


def count_params(model) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_run_base_dir(result_dir: str, tag: str) -> str:
    """
    Generate a base directory for each experiment run.
    Looks like this: {result_dir}/{date}_{suffix}/{tag}
    Args:
        result_dir (str): Experiment output directory.
        tag (str): Experiment tag.

    Returns:
        str: Directory name.
    """
    timestamp = time()
    date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%y%m%d_%H%M")
    base_dir = os.path.join(result_dir, f"{date_str}_{tag}") + "/"
    ensure_dir(base_dir)
    return base_dir


def ensure_dir(d):
    """Ensure that a directory exists"""
    # Create result dir on the fly
    if not os.path.exists(d):
        os.makedirs(d)


def parse_args():
    """
    Define and parse commandline arguments.
    """
    # training settings
    parser = argparse.ArgumentParser(description="Food Interpolator Experiment")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--data-dir", help="path to the result directory", metavar="DIR", required=True
    )
    parser.add_argument(
        "--restore", help="path to the restore directory", metavar="DIR"
    )
    parser.add_argument(
        "--result-dir",
        default="results",
        help="path to the result directory",
        metavar="DIR",
    )
    parser.add_argument(
        "--dim", type=int, default=64, metavar="N", help="Model dimensionality."
    )
    parser.add_argument(
        "--critic-iters",
        type=int,
        default=5,
        metavar="N",
        help="How may iterations to train the critic for.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        metavar="N",
        help="How may iterations to train the critic for.",
    )
    parser.add_argument(
        "--log-iter", type=int, default=100, metavar="N", help="Log after N iterations."
    )
    parser.add_argument(
        "--lamb",
        type=int,
        default=10,
        metavar="N",
        help="Gradient penalty lambda hyperparameter.",
    )
    parser.add_argument(
        "--acgan-scale-c",
        type=float,
        default=1.0,
        help="How to scale the critic's ACGAN loss relative to WGAN loss.",
    )
    parser.add_argument(
        "--acgan-scale-g",
        type=float,
        default=1.0,
        help="How to scale the generators's ACGAN loss relative to WGAN loss.",
    )
    parser.add_argument(
        "--generator-iters",
        type=int,
        default=1,
        metavar="N",
        help="How may iterations to train the generator for.",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enable CUDA training"
    )
    parser.add_argument(
        "--cuda-device-id",
        nargs="+",
        type=int,
        default=[0],
        help="Cuda device ids. E.g. [0,1,2]. Use -1 for all GPUs available and -2 for cpu only.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debugging."
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="Tag to identify runs in the result directory and tensorboard overviews",
        required=True,
    )
    parser.add_argument(
        "--force-overfit",
        action="store_true",
        default=False,
        help="Force overfitting (set num train samples to 1000)",
    )
    parser.add_argument(
        "--make-interpolations",
        action="store_true",
        default=False,
        help="Flag if this run should only generate interpolations. If true, --restore flag must be set.",
    )

    parser.add_argument("--interpolation-steps", help="Number of interpolation steps.")

    args = parser.parse_args()
    ensure_dir(args.result_dir)

    if args.debug:
        args.epochs = 2
        args.batch_size = 10

    return args


def print_args(args):
    """Print all experiment arguments."""
    logger.info("Experiment started with the following arguments:")
    for key, value in sorted(vars(args).items(), key=lambda x: x[0]):
        logger.info(f"{key: <15} {value}")


# import sklearn.datasets

args = parse_args()
torch.manual_seed(args.seed)

print_args(args)

DATA_DIR = args.data_dir
VAL_DIR = DATA_DIR

IMAGE_DATA_SET = (
    "custom"
)  # change this to something else, e.g. 'imagenets' or 'raw' if your data is just a folder of raw images.
# If you use lmdb, you'll need to write the loader by yourself, see load_data
# TRAINING_CLASS = os.listdir(DATA_DIR)
TRAINING_CLASS = ["pizza", "burger"]
VAL_CLASS = TRAINING_CLASS
NUM_CLASSES = len(VAL_CLASS)

if len(DATA_DIR) == 0:
    raise Exception("Please specify path to data directory in gan_64x64.py!")

RESTORE_MODE = (
    args.restore
)  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0  # starting iteration
# MODE = 'wgan-gp'
DIM = args.dim  # Model dimensionality
CRITIC_ITERS = args.critic_iters  # How many iterations to train the critic for
GENER_ITERS = args.generator_iters
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = args.batch_size  # Batch size. Must be a multiple of N_GPUS
END_ITER = args.iterations  # How many iterations to train for
LAMBDA = args.lamb  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = args.dim * args.dim * 3  # Number of pixels in each iamge
ACGAN_SCALE = (
    args.acgan_scale_c
)  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = (
    args.acgan_scale_g
)  # How to scale generator's ACGAN loss relative to WGAN loss
LOG_ITER = args.log_iter
set_dims(DIM, OUTPUT_DIM)


OUTPUT_PATH = generate_run_base_dir(
    args.result_dir, args.tag
)  # output path where result (.e.g drawing images, cost, chart) will be stored
setup_logging(os.path.join(OUTPUT_PATH, "log.txt"))


def showMemoryUsage(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    logger.info(
        "Used/total: " + "{}/{}".format(item["memory.used"], item["memory.total"])
    )


def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def plot_sample(x, y):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x: Image.
        y: Target.
        y_pred: Target prediction.
        loss: Loss value.
    """
    import matplotlib.pyplot as plt

    # x = (x - x.min()) / (x.max() - x.min())
    x = x * data_stds.view(1, 3, 1, 1) + data_means.view(1, 3, 1, 1)
    tensors = torchvision.utils.make_grid(x, nrow=8, padding=1)
    plt.imshow(tensors.permute(1, 2, 0))
    plt.title("y={}".format(y.squeeze().numpy()))
    plt.show()


def load_data(path_to_folder, classes, batch_size):
    """load the dataset."""
    # Data transformations
    data_transform = transforms.Compose(
        [
            transforms.Resize((args.dim, args.dim)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_means.cpu().numpy(), std=data_stds.cpu().numpy()
            ),
        ]
    )

    classes = set(classes)

    def filter_data(path):
        """Filter files by selected classes."""
        # conditions_and = []
        # conditions_or = [True]
        # conditions_and.append(path.split("/")[-2] in classes)

        # conditions_or.append(path.endswith("png"))
        # conditions_or.append(path.endswith("jpg"))
        # conditions_or.append(path.endswith("jpeg"))

        return path.split("/")[-2] in classes and (
            path.endswith("png") or path.endswith("jpg") or path.endswith("jpeg")
        )

    # Define dataset via imagefolder
    dataset = datasets.ImageFolder(
        root=path_to_folder, transform=data_transform  # , is_valid_file=filter_data
    )
    ####### {{{
    # Custom code copied from datasets.folder.DatasetFolder to support partial selection of classes
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

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        drop_last=True,
        pin_memory=True,
    )
    return dataset_loader


def compute_means_stds(train_loader):
    """Compute the means and standard deviations of each channel."""
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    print("start computing means")
    for data, target in train_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        print(nb_samples / len(train_loader.dataset) * 100)

    mean /= nb_samples
    std /= nb_samples
    print("Means:", mean)
    print("Stds:", std)
    exit()


def calc_gradient_penalty(netD, real_data, fake_data):
    # Mixing factor between real and fake data
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, 3 * DIM * DIM).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)

    # Interpolate between real and fake data
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)

    # Get gradients
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_image(netG, noise=None):
    if noise is None:
        rand_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        noise = gen_rand_noise_with_label(rand_label)
    with torch.no_grad():
        noisev = noise.to(device)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 3, DIM, DIM)

    # Remove normalization
    samples = samples * data_stds.view(1, 3, 1, 1) + data_means.view(1, 3, 1, 1)

    return samples


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
    return v / norm.view(-1, 1)


def make_interpolation(netG):
    """
    Generate interpolations.

    Args:
        netG: Generator network.
    """
    # Generate samples with class=0
    noise_start_label = np.zeros(BATCH_SIZE, dtype=int)
    noise_start = gen_rand_noise_with_label(noise_start_label)

    # Generate samples with class=1
    noise_end_label = np.ones(BATCH_SIZE, dtype=int)
    noise_end = gen_rand_noise_with_label(noise_end_label)

    # Alpha from 0 to 1 in 100 steps
    for alpha in np.linspace(0, 1, num=args.interpolation_steps, endpoint=True):
        noise = interp_circular(noise_start, noise_end, alpha)

        yield generate_image(netG, noise)


def gen_rand_noise_with_label(label=None):
    if label is None:
        label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    # attach label into noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, 128))
    prefix = np.zeros((BATCH_SIZE, NUM_CLASSES))
    prefix[np.arange(BATCH_SIZE), label] = 1
    noise[np.arange(BATCH_SIZE), :NUM_CLASSES] = prefix[np.arange(BATCH_SIZE)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

data_means = torch.tensor([0.4494, 0.4235, 0.4196]).to(device)
data_stds = torch.tensor([0.9628, 0.9711, 1.0579]).to(device)

fixed_label = []
for c in range(BATCH_SIZE):
    fixed_label.append(c % NUM_CLASSES)
fixed_noise = gen_rand_noise_with_label(fixed_label)

if RESTORE_MODE:
    aG = torch.load(os.path.join(args.restore, "generator.pt"))
    aD = torch.load(os.path.join(args.restore, "discriminator.pt"))
else:
    aG = GoodGenerator(DIM, DIM * DIM * 3)
    aD = GoodDiscriminator(DIM, NUM_CLASSES)

    aG.apply(weights_init)
    aD.apply(weights_init)

print("Generator #params:    ", count_params(aG))
for name, param in aG.named_modules():
    print(">>>>>  ", name, ":    ", count_params(param))
print("Discriminator #params:", count_params(aD))
for name, param in aD.named_modules():
    print(">>>>>  ", name, ":    ", count_params(param))

# Create interpolations
if args.make_interpolations:
    aG = aG.to(device)
    for i, samples in enumerate(make_interpolation(aG)):
        print("Interpolation", i, "of", args.interpolation_steps)
        torchvision.utils.save_image(
            samples, OUTPUT_PATH + "samples_{:06}.png".format(i), nrow=8, padding=2
        )
    exit()


LR = 2e-4

optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0.5, 0.9))
schedule = lambda iter: (END_ITER - iter) / END_ITER * LR
scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, schedule)
scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, schedule)

aux_criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()

aG = aG.to(device)
aD = aD.to(device)
aG = make_multi_gpu(aG)
aD = make_multi_gpu(aD)

print(80 * "=")
print("Generator")
print(aG)
print(80 * "=")
print("Discriminator")
print(aD)


writer = SummaryWriter()
# Reference: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
def train():
    dataloader = load_data(DATA_DIR, TRAINING_CLASS, BATCH_SIZE)
    dataiter = iter(dataloader)
    batch = next(dataiter)
    for iteration in tqdm(range(START_ITER, END_ITER)):
        start_time = time()
        # ---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
            aG.zero_grad()
            f_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            noise = gen_rand_noise_with_label(f_label)
            noise.requires_grad_(True)
            fake_data = aG(noise)
            gen_cost, gen_aux_output = aD(fake_data)

            aux_label = torch.from_numpy(f_label).long()
            aux_label = aux_label.to(device)
            aux_errG = aux_criterion(gen_aux_output, aux_label).mean()
            gen_cost = -gen_cost.mean()
            g_cost = ACGAN_SCALE_G * aux_errG + gen_cost
            g_cost.backward()

        optimizer_g.step()
        # ---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):

            aD.zero_grad()

            # gen fake data and load real data
            f_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            noise = gen_rand_noise_with_label(f_label)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            real_data = batch[0]  # batch[1] contains labels
            real_data.requires_grad_(True)
            real_label = batch[1]
            # logger.info("r_label" + str(r_label))

            real_data = real_data.to(device)
            real_label = real_label.to(device)

            # train with real data
            disc_real, aux_output = aD(real_data)
            aux_errD_real = aux_criterion(aux_output, real_label)
            errD_real = aux_errD_real.mean()
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake, aux_output = aD(fake_data)
            # aux_errD_fake = aux_criterion(aux_output, fake_label)
            # errD_fake = aux_errD_fake.mean()
            disc_fake = disc_fake.mean()

            # showMemoryUsage(0)
            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)
            # showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_acgan = errD_real  # + errD_fake
            (disc_cost + ACGAN_SCALE * disc_acgan).backward()
            w_dist = disc_fake - disc_real
            optimizer_d.step()
            # ------------------VISUALIZATION----------
            # if i == CRITIC_ITERS - 1:
            #     writer.add_scalar("data/disc_cost", disc_cost, iteration)
            #     # writer.add_scalar('data/disc_fake', disc_fake, iteration)
            #     # writer.add_scalar('data/disc_real', disc_real, iteration)
            #     writer.add_scalar("data/gradient_pen", gradient_penalty, iteration)
            #     writer.add_scalar("data/ac_disc_cost", disc_acgan, iteration)
            #     writer.add_scalar("data/ac_gen_cost", aux_errG, iteration)
            #     # writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
            #     # writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
            #     # writer.add_scalar('data/fake_data_mean', fake_data.mean())
            #     # writer.add_scalar('data/real_data_mean', real_data.mean())
            #     # if iteration %200==99:
            #     #    paramsD = aD.named_parameters()
            #     #    for name, pD in paramsD:
            #     #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
            #     if iteration % LOG_ITER == (LOG_ITER - 1):
            #         if type(aD) == torch.nn.DataParallel:
            #             body_model = [i for i in aD.children()][0].conv1
            #         else:
            #             body_model = aD.conv1
            #         layer1 = body_model.conv
            #         xyz = layer1.weight.data.clone()

            #         tensor = xyz.cpu()
            #         tensors = torchvision.utils.make_grid(tensor, nrow=8, padding=1)
            #         writer.add_image("D/conv1", tensors, iteration)

        # ---------------VISUALIZATION---------------------
        writer.add_scalar("data/gen_cost", gen_cost, iteration)
        # if iteration %200==199:
        #   paramsG = aG.named_parameters()
        #   for name, pG in paramsG:
        #       writer.add_histogram('G.' + name, pG.clone().data.cpu().numpy(), iteration)
        # ----------------------Generate images-----------------

        lib.plot.plot(OUTPUT_PATH + "time", time() - start_time)
        lib.plot.plot(OUTPUT_PATH + "train_disc_cost", disc_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + "train_gen_cost", gen_cost.cpu().data.numpy())
        lib.plot.plot(OUTPUT_PATH + "wasserstein_distance", w_dist.cpu().data.numpy())
        if iteration % LOG_ITER == (LOG_ITER - 1):
            val_loader = load_data(VAL_DIR, VAL_CLASS, BATCH_SIZE)
            dev_disc_costs = []
            for _, images in enumerate(val_loader):
                if i > 10:
                    break
                imgs = torch.Tensor(images[0])
                imgs = imgs.to(device)
                with torch.no_grad():
                    imgs_v = imgs

                D, _ = aD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot(OUTPUT_PATH + "dev_disc_cost.png", np.mean(dev_disc_costs))
            lib.plot.flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(
                gen_images,
                OUTPUT_PATH + "samples_{:06}.png".format(iteration),
                nrow=8,
                padding=2,
            )
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image("images", grid_images, iteration)
            # gen_images = generate_image(iteration, aG, persistant_noise)
            # gen_images = torchvision.utils.make_grid(torch.from_numpy(gen_images), nrow=8, padding=1)
            # writer.add_image('images', gen_images, iteration)
            # ----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        lib.plot.tick()

        # Reduce learning rate
        scheduler_g.step()
        scheduler_d.step()


train()
