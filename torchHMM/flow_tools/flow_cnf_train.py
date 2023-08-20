import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_moons


from cnf_utils import build_model_tabular, standard_normal_logprob


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--example', default="2", required=False, help='Example (1 or 2, default: %(default)s)')
    parser.add_argument('--n_train', default="2000", required=False, help='nr of training data (default: %(default)s)')
    parser.add_argument('--nr_epochs', default="1000", required=False, help='nr of epochs (default: %(default)s)')
    parser.add_argument('--lrate', default="0.001", required=False, help='learning rate (default: %(default)s)')
    parser.add_argument('--dims', default="256", required=False, help='dims (default: %(default)s)')
    parser.add_argument('--output-model', default="flow_model.pt", required=False,
                        help='output model (default: %(default)s)')
    parser.add_argument('--output-train-data', default="train_data.pkl", required=False,
                        help='train data  (default: %(default)s)')

    args = parser.parse_args()

    return args.n_train, args.nr_epochs, args.lrate, args.example, args.output_model, args.output_train_data, args.dims



def ParseFlowArgs(command_args: Optional[list[str]] = None):
    """
    Parse arguments for flow experiments.

    Args:
        command_args: list of command line arguments to parse. If None, parse sys.argv.
        Example: ["--seed", "117", "--lrate", "0.01"]

    Returns:

    """
    if command_args is None:
        command_args = []

    NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
    SOLVERS = [
        "dopri5",
        "bdf",
        "rk4",
        "midpoint",
        "adams",
        "explicit_adams",
        "fixed_adams",
    ]
    LAYERS = [
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ]

    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument(
        "--seed",
        default=117,
        type=int,
        required=False,
        help="default seed"
        #    "--seed", default=116, type=int, required=False, help="default seed"
    )
    parser.add_argument("--lrate", default="0.01", required=False, help="learning rate")
    parser.add_argument(
        "--layer_type",
        type=str,
        default="concatsquash",
        choices=LAYERS,
    )
    parser.add_argument("--dims", type=str, default="16-16")
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of stacked CNFs."
    )
    parser.add_argument("--time_length", type=float, default=0.5)
    parser.add_argument("--train_T", type=eval, default=True)
    parser.add_argument("--add_noise", type=eval, default=False, choices=[True, False])
    parser.add_argument("--noise_var", type=float, default=0.1)
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="brute_force",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES
    )

    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--spectral_norm", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument("--batch_norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--bn_lag", type=float, default=0)
    parser.add_argument(
        "--max_shape",
        type=int,
        default=1000,
        required=False,
        help="max number of samples used when training EM",
    )
    args = parser.parse_args(command_args)
    return args


def train_flow(cnf, data, nr_epochs=500, lr=0.001, weight_decay=0.00001):
    optimizer = torch.optim.Adam(cnf.parameters(), lr=lr, weight_decay=weight_decay)
    for k in range(nr_epochs):
        optimizer.zero_grad()
        y, delta_log_py = cnf(data, torch.zeros(data.size(0), 1).to(data))
        log_py = standard_normal_logprob(y).sum(1)
        delta_log_py = delta_log_py.sum(1)
        log_px = log_py - delta_log_py
        log_px = -log_px.mean()
        log_px.backward()
        optimizer.step()
        loss_numpy = log_px.cpu().detach().numpy()
        print(
            "Epoch = ",
            k,
            "/",
            nr_epochs,
            ",\t loss: ",
            np.round(loss_numpy, 6),
        )


def sample_flow(cnf, size, device):
    y = torch.randn(*size).float().to(device)
    x = cnf(y, None, reverse=True).view(*y.size())
    return x


def sample_data_2D(n_samples) -> np.ndarray:
    """
    Sample 2D dataset
    Returns: numpy array of the shape (n_samples, 2).
    """
    u1 = np.random.uniform(0.0, 1.0, n_samples)
    u2 = np.random.uniform(0.0, 1.0, n_samples)
    x = -np.log(u1)
    y = -np.log(u2) / x
    result = np.array([x, y]).T
    return result



def sample_data_2D_moons(n_samples) -> np.ndarray:
    """
    Sample 2D dataset: MOONS
    Returns: numpy array of the shape (n_samples, 2).
    """

    points, classes_true = make_moons(n_samples, noise=.05, random_state=0)

    x = points[:, 0]
    y = points[:, 1]


    result = np.array([x, y]).T
    return result



def sample_data_1D(samples):
    n_samples1, n_samples2, n_samples3=   int(samples / 4), int(samples / 2), int(samples/ 4)


    x = np.random.beta(7, 1.1, size=(n_samples1, 1))
    y = np.random.uniform(low=0.2, high=0.4, size=(n_samples2, 1))
    z = np.random.normal(loc=0.6, scale=0.06782329983125268, size=(n_samples3, 1))
    return np.concatenate([x, y, z])







#main

n_train, nr_epochs, lrate, example, output_model, output_train_data, dims = ParseArguments()

n_train = int(n_train)
nr_epochs = int(nr_epochs)
lrate = float(lrate)
example = int(example)

#print(n_train, nr_epochs, lrate, example, output_model, output_train_data)


args = ParseFlowArgs(["--dims", f"{dims}-{dims}"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if(example==1):
    samples = sample_data_1D(n_train)

if(example==2):
    samples = sample_data_2D_moons(n_train)

output_train_data_file = output_train_data.split(".pkl")[0]+"_n_"+str(n_train)+".pkl"

with open(output_train_data_file, "wb") as file:
    pickle.dump(samples, file)

print("DDDFD ",output_train_data_file)
quit()

data_dim = samples.shape[1]

flow_model = build_model_tabular(args, samples.shape[1]).to(device)

samples_torch = torch.tensor(samples).float().to(device)
print("Training model:")
train_flow(flow_model, samples_torch, nr_epochs=nr_epochs, lr=lrate, weight_decay=0.00001)
print("Done (training)")


torch.save(flow_model,output_model)

print("Saved training data as:\t ", output_train_data_file)
print("Saved flow model as as:\t ",output_model)

