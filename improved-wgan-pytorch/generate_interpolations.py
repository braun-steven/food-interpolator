import torch
import numpy as np
import argparse


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
    norm = alpha * np.linalg.norm(v1) + (1 - alpha) * np.linalg.norm(v2)
    return v / norm


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
        help="How may iterations to train the critic for.",
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

    args = parser.parse_args()
    ensure_dir(args.result_dir)

    if args.debug:
        args.epochs = 2
        args.batch_size = 10

    return args


args = parse_args()
BATCH_SIZE = 64
NUM_CLASSES = 2


def make_interpolation(netG):
    """
    Generate interpolations.

    Args:
        netG: Generator network.
    """
    # Generate samples with class=0
    noise_start_label = np.random.zeros(BATCH_SIZE)
    noise_start = gen_rand_noise_with_label(noise_start_label)

    # Generate samples with class=1
    noise_end_label = np.random.ones(BATCH_SIZE)
    noise_end = gen_rand_noise_with_label(noise_end_label)

    # Alpha from 0 to 1 in 100 steps
    results = []
    for alpha in np.linspace(0, 1, num=100, endpoint=True):
        noise = interp_circular(noise_start, noise_end, alpha)

        fake_data = generate_image(netG, noise)
        results.append(fake_data)

    return results


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

aG = torch.load(os.path.join(args.restore, "generator.pt"))
aD = torch.load(os.path.join(args.restore, "discriminator.pt"))

if args.make_interpolations:
    interps = make_interpolation(aG)
    for i, samples in enumerate(interps):
        torchvision.utils.save_image(
            samples, OUTPUT_PATH + "samples_{:06}.png".format(i), nrow=8, padding=2
        )
    exit()
