from hmmlearn import hmm
from hmmlearn.base import _log
import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
import torch
from typing import Optional
from math import prod
from torchHMM.flow_tools.cnf_utils import standard_normal_logprob, build_model_tabular
import itertools
import wandb

# czy jest hmm w torchu już zaimplementowany:
# http://torch.ch/torch3/manual/HMM.html
# https://github.com/nwams/Hidden_Markov_Model-in-PyTorch
# could be useful also: https://pyro4ci.readthedocs.io/en/latest/_modules/pyro/distributions/hmm.html

# TODO: recheck cavariance learning in HmmOptim!
# TODO: recheck Latin cube
# TODO: add custom covergence  monitor

# Future features:
# try reading data out of file when using quasi random nodes
# torch model with embeddings in separate class



DISCRETIZATION_TECHNIQUES = ["grid", "random", "latin_cube_u", "latin_cube_q", "uniform", "sobol",  "halton"]
OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)
LEARNING_ALGORITHMS = ["em", "em_dense", "cooc"]


class Args():
    def __init__(self,  args_dict):

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
        self.args_dict = args_dict

        self.seed = self.set_value('seed',  2023)
        self.lrate = self.set_value('lrate', 0.01)
        self.layer_type = self.set_value('layer_type', "concatsquash")
        self.dims = self.set_value('dims', "16-16")
        self.num_blocks = self.set_value("num_blocks", 2)
        self.time_length = self.set_value('time_length', 0.5)
        self.train_T = self.set_value('train_T', True)
        self.add_noise = self.set_value('add_noise', False)
        self.noise_var = self.set_value('noise_var',  0.1)
        self.divergence_fn = self.set_value('divergence_fn', "brute_force")
        self.nonlinearity = self.set_value("nonlinearity", "tanh")
        self.solver = self.set_value('solver', "dopri5")
        self.atol = self.set_value('atol', 1e-5)
        self.rtol = self.set_value('atol', 1e-5)
        self.step_size = self.set_value('step_size', None)
        self.test_solver = self.set_value('test_solver', None)
        self.test_atol = self.set_value('test_atol', None)
        self.test_rtol = self.set_value('test_rtol', None)
        self.residual = self.set_value('residual', False)
        self.rademacher = self.set_value('rademacher', False)
        self.spectral_norm = self.set_value('spectral_norm', False)
        self.batch_norm = self.set_value('batch_norm', True)
        self.bn_lag = self.set_value('bn_lag', 0)
        self.max_shape = self.set_value('max_shape', 1000)

    def set_value(self, key, default_val):
        if key in self.args_dict.keys():
            return self.args_dict[key]
        else: return default_val

class FlowHmmOptim(torch.nn.Module):
    def __init__(
        self,
        n_components: int,
        n_dim: int,
        startprob_: Optional[npt.NDArray] = None,
        transmat_: Optional[npt.NDArray] = None,  # Initial values
        trainable: str = "",
        trans_from: str = "S",
        args = dict()
    ):
        """
        Initialize torch.nn.Module for HMM parameters estimation
        :param n_components: number of hidden states
        :param n_dim: dimensionality of observations
        :param means_: initial value for means
        :param covars_: initial value for cavariances
        :param startprob_: initial value for starting probability
        :param transmat_: initial value for transition matrix
        :param trainable: string containing codes for parameters that need estimation
        """
        super(FlowHmmOptim, self).__init__()

        transmat = np.abs(
            transmat_
            if transmat_ is not None
            else np.random.standard_normal(n_components * n_components).reshape(
                n_components, n_components
            )
        )
        transmat /= transmat.sum(axis=1)[:, np.newaxis]

        startprob = np.abs(
            startprob_
            if startprob_ is not None
            else np.random.standard_normal(n_components)
        )
        startprob /= startprob.sum()

        self.n_components = n_components
        self.n_dim = n_dim
        self.trainable = trainable

        self._S_unconstrained = torch.nn.Parameter(
            torch.tensor(np.log(transmat * startprob[:, np.newaxis])), requires_grad="t" in trainable
        )

        self.NFs = torch.nn.ModuleList([build_model_tabular(Args(args), n_dim) for _ in range(n_components)])

    def emission_score(self, cnf, nodes):
        y, delta_log_py = cnf(nodes, torch.zeros(nodes.size(0), 1).to(nodes))
        log_py = standard_normal_logprob(y).sum(1)
        delta_log_py = delta_log_py.sum(1)
        log_px = log_py - delta_log_py
        return log_px
    def forward(self, nodes):
        """
        Calculate the forward pass of the torch.nn.Module
        :return: cooc matrix from current parameters
        """

        B = torch.nn.functional.normalize(
            torch.cat(
                [
                    torch.exp(self.emission_score(dist_model, nodes)).reshape(
                        1, -1
                    )
                    for dist_model in self.NFs
                ],
                dim=0,
            ),
            dim=1, p=1
        )

        S_ = torch.exp(self._S_unconstrained)
        S = S_ / S_.sum()
        return (B.T.double() @ S.double() @ B.double()).float()

    @staticmethod
    def _to_numpy(tens: torch.tensor):
        """
        Get value of torch tensor as a numpy array
        :param tens: torch tensor (or parameter)
        :return: numpy array
        """
        return tens.clone().numpy(force=True) #.clone().cpu().detach().numpy()  # TODO: check if it will be working in all cases

    def get_model_params(self):
        """
        Retrieve HMM parameters from torch.nn.Module
        :return: means, covars, transmat, startprob
        """
        # TODO: https://github.com/tooploox/flowhmm/blob/main/src/flowhmm/models/fhmm.py linijka 336 - czy mi to potrzebne
        S_ = torch.exp(self._S_unconstrained)
        S = S_ /S_.sum()
        if S.sum() == 0:
            S = torch.ones(S.shape)
            S = S / S.sum(axis=1).view(-1, 1)
        startprob = torch.sum(S, dim=1)
        transmat = S / startprob.unsqueeze(1)

        return (
            self.NFs,
            self._to_numpy(transmat),
            self._to_numpy(startprob),
        )


class FlowHMM(hmm.CategoricalHMM):
    def __init__(
        self,
        discretization_method: str = "random",
        no_nodes: int = 100,
        l: Optional[int] = None,
        learning_alg: str = "cooc",
        n_components: int = 1,
        startprob_prior: float = 1.0,
        transmat_prior: float = 1.0,
        optim_params: Optional[str] = None,
        optimizer: str = "SGD",

        algorithm: str = "viterbi",
        random_state: int = None,
        n_iter: int = 10,
        tol: float = 0.01,
        verbose: bool = False,
        params: str = "te",  # TODO: default without 's'
        init_params: str = "te",
        implementation: str = "log",
    ) -> None:
        super(FlowHMM, self).__init__(
            n_components=n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )

        assert (
            discretization_method.lower() in DISCRETIZATION_TECHNIQUES
        ), f"discretization method: '{discretization_method}' not allowed, choose one of {DISCRETIZATION_TECHNIQUES}"

        self.discretization_method = discretization_method.lower()
        self.no_nodes = no_nodes
        self.nodes = None  # Placeholder

        self.learning_alg = learning_alg

        if self.learning_alg in ["cooc"]:  # TODO: update in further development
            self.max_epoch = (
                optim_params.pop("max_epoch", 10000)
                if optim_params is not None
                else 10000
            )
            self.model = None
            try:
                self.optimizer = eval(f"torch.optim.{optimizer}")
            except:
                _log.warning(
                    f"Optimizer not found: {optimizer}. SGD optimizer will be used instead"
                )
                self.optimizer = torch.optim.SGD
            self.optim_params = (
                optim_params if optim_params is not None else dict(lr=0.001)
            )

        # TODO: make it optional
        self.l = l
        self.z_, self.u_ = None, None

    def _provide_nodes_grid(self, X: npt.NDArray):
        """
        Select random observations as nodes for discretization; nodes are saved in attribute nodes
        :param X: Original, continuous (gaussian) data
        """
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        dims = [1]
        for i in range(X.shape[1]):
            dims.append(int((self.no_nodes / dims[i]) ** (1 / (X.shape[1] - i))))
        grids = [np.linspace(mins[i], maxs[i], dims[i + 1]) for i in range(X.shape[1])]
        self.nodes = np.vstack([np.array([grids[d][ind[d]] for d in range(X.shape[1])]) for ind in itertools.product(*[[i for i in range(r)] for r in dims[1:]])]).T
        


    def _provide_nodes_random(self, X: npt.NDArray):
        """
        Select random observations as nodes for discretization; nodes are saved in attribute nodes
        Works for any dimension
        :param X: Original, continuous (gaussian) data
        """
        self.nodes = X[
            np.random.choice(X.shape[0], size=self.no_nodes, replace=False)
        ].transpose()

    def _provide_nodes_latin_q(self, X: npt.NDArray):
        """
        Provide nodes from CDF on latin qube; nodes are saved in attribute nodes
        Works for any dimension
        :param X: Original, continuous (gaussian) data
        """
        self.nodes = np.apply_along_axis(
            lambda x: np.quantile(x[: (-self.no_nodes)], x[(-self.no_nodes):]),
            0,
            np.concatenate(
                [X, qmc.LatinHypercube(X.shape[1]).random(self.no_nodes)],
                axis=0,
            ),
        ).T

    def _provide_nodes_sobol(self, X: npt.NDArray):  # each point in a row
        """
        Provide nodes from a latin qube on cuboid of observations; nodes are saved in attribute nodes
        :param X:  Original, continuous (gaussian) data
        """
        self.nodes = (
            qmc.Sobol(X.shape[1]).random(self.no_nodes)
            * (X.max(axis=0) - X.min(axis=0))[np.newaxis, :]
            + X.min(axis=0)[np.newaxis, :]
        ).T

    def _provide_nodes_halton(self, X: npt.NDArray):  # each point in a row
        """
        Provide nodes from a latin qube on cuboid of observations; nodes are saved in attribute nodes
        :param X:  Original, continuous (gaussian) data
        """
        self.nodes = (
            qmc.Halton(X.shape[1]).random(self.no_nodes)
            * (X.max(axis=0) - X.min(axis=0))[np.newaxis, :]
            + X.min(axis=0)[np.newaxis, :]
        ).T

    def _provide_nodes_latin_u(self, X: npt.NDArray):  # each point in a row
        """
        Provide nodes from a latin qube on cuboid of observations; nodes are saved in attribute nodes
        :param X:  Original, continuous (gaussian) data
        """
        self.nodes = (
            qmc.LatinHypercube(X.shape[1]).random(self.no_nodes)
            * (X.max(axis=0) - X.min(axis=0))[np.newaxis, :]
            + X.min(axis=0)[np.newaxis, :]
        ).T

    def _provide_nodes_uniform(self, X: npt.NDArray):
        """
        Provide nodes uniformly distributed on cuboid of observations; nodes are saved in attribute nodes
        :param X: Original, continuous (gaussian) data
        """
        self.nodes = (
            np.random.uniform(size=self.no_nodes * X.shape[1]).reshape(
                self.no_nodes, X.shape[1]
            )
            * (X.max(axis=0) - X.min(axis=0))[np.newaxis, :]
            + X.min(axis=0)[np.newaxis, :]
        ).transpose()

    def provide_nodes(self, X: npt.NDArray, force: bool):
        """
        Provide nodes for discretization according to models discretization method; nodes are saved in attribute nodes
        :param X: Original, continuous (gaussian) data
        :param force: If nodes should be updated, when they have been previously specified
        """
        if not force and (self.nodes is not None):
            if self.verbose:
                print("Nodes have been already set. Use force=True to update them")
            pass
        elif self.discretization_method == "random":
            self._provide_nodes_random(X)
        elif self.discretization_method == "grid":
            self._provide_nodes_grid(X)
        elif self.discretization_method == "latin_cube_q":
            self._provide_nodes_latin_q(X)
        elif self.discretization_method == "latin_cube_u":
            self._provide_nodes_latin_u(X)
        elif self.discretization_method == "sobol":
            self._provide_nodes_sobol(X)
        elif self.discretization_method == "halton":
            self._provide_nodes_halton(X)
        else:
            self._provide_nodes_uniform(X)
        self.n_features = self.nodes.shape[1]

    def discretize(self, X: npt.NDArray, force: bool):
        """
        Provide nodes for discretization and represent continuous data as cluster indexes
        :param X: Original, continuous (gaussian) data
        :param force: Should nodes be updated, if they are already provided.
        :return: Discretized data (index of cluster)
        """
        self.provide_nodes(X, force)
        res = np.array([])
        batchsize = 1000
        for i in range((X.shape[0] // batchsize) + 1):
            res = np.concatenate([res, np.argmin(  # TODO:  fix this!
                np.square(X[(i * batchsize):((i + 1) * batchsize), :, np.newaxis] - self.nodes[np.newaxis, :, :]).sum(
                    axis=1),
                axis=1,
            ).reshape(-1)]).astype(int)
        return res


    def _needs_init(self, code: str, name: str, torch_check: bool = False):
        """
        Decide wether the attribute needs to be initialized (based on model setup)
        :param code: short code possibly included in init params
        :param name: name of sttributed to initialize
        :param torch_check: is the check provided for torch initial values (True) or HMM initial values (False)
        :return: boolean, should attribute be initialized
        """
        if torch_check:
            result = code in self.init_params
            if code == "t":
                if "s" in self.init_params:
                    _log.warning(
                        "Optimizing separately attribute 'startprob_' ignores the stationarity requirement"
                    )
            if code in ["t", "s"]:
                if "z" in self.init_params and "u" in self.init_params:
                    _log.warning(
                        "Attributes 'startprob_' and 'transmat_' will be initialized based on "
                        "attributes 'u_' and 'z_'"
                    )
                    result = False
            if code in ["z", "u"]:
                result = (code in self.params) and (code in self.init_params)
        else:
            result = super()._needs_init(code, name)
        return result

    @staticmethod
    def compute_stationary(
        matrix,
    ):  # TODO: or maybe it should be a separate function -- think of it
        """
        Compute stationary distiobution of a stochastic matrix
        :param matrix: stochastic matrix
        :return: vector stationary distribution
        """
        vals, vecs = np.linalg.eig(matrix.T)
        vec1 = vecs[:, np.isclose(vals, 1)].reshape(-1)
        stationary = vec1 / vec1.sum()
        return stationary.real

    def _init(self, X: npt.NDArray, lengths: Optional[npt.NDArray[int]] = None):
        """
        Initialize model parameters and prepare data before training
        :param X: Original, continuous (gaussian) data
        :param lengths: Lengths of individual sequences in X
        """
        # init k-means with a batch of data (of some maximum size)?
        super()._init(self.discretize(X, False).reshape(-1, 1))
        for e in ["z", "u"]:
            if self._needs_init(e, f"{e}_"):
                setattr(
                    self,
                    f"{e}_",
                    np.random.standard_normal(self.l * self.n_components).reshape(
                        self.l, self.n_components
                    ),
                )

        torch_inits = dict(
            n_components=self.n_components, n_dim=X.shape[1], trainable=self.params
        )

        # TODO: check if placeholders are needed
        # torch_inits["means_"] = self.means_
        # torch_inits["covars_"] = self.covars_
        # if self._needs_init("z", "z_", True) and self._needs_init("u", "u_", True):
        #     torch_inits["z_"] = self.z_
        #     torch_inits["u_"] = self.u_
        torch_inits["startprob_"] = self.startprob_
        torch_inits["transmat_"] = self.transmat_
        if self.learning_alg == "cooc":
            self.model = FlowHmmOptim(**torch_inits)

    def _fit_em_dense(
        self, X: npt.NDArray, lengths: Optional[npt.NDArray[int]] = None
    ):  # TODO: add for Gaussian Dense HMMs
        """
        TODO
        :param X:
        :param lengths:
        :return:
        """
        pass

    def _cooccurence(
        self, Xd: npt.NDArray[int], lengths: Optional[npt.NDArray[int]] = None
    ):
        """
        Process discrete data sequences into co-occurrence matrix
        :param Xd: Disretized data (represented as cluster indexes)
        :param lengths: Lengths of individual sequences in X
        :return: co-occurrence matrix
        """
        # TODO: https://github.com/tooploox/flowhmm/blob/main/src/flowhmm/models/fhmm.py linijka 88 - skąd taki dzielnik??
        cooc_matrix = np.zeros(shape=(self.nodes.shape[1], self.nodes.shape[1]))
        cont_seq_ind = np.ones(shape=Xd.shape[0])
        if lengths is None:
            lengths = np.array([Xd.shape[0]])
        cont_seq_ind[lengths.cumsum() - 1] *= 0
        for i in range(Xd.shape[0] - 1):
            cooc_matrix[Xd[i], Xd[i + 1]] += cont_seq_ind[i]
        cooc_matrix /= cooc_matrix.sum()
        return cooc_matrix

    def _fit_cooc(
        self,
        Xd: npt.NDArray,
        Xc: Optional[npt.NDArray],
        lengthsd: Optional[npt.NDArray[int]] = None,
        lengthsc: Optional[npt.NDArray[int]] = None,
        early_stopping: bool = False
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        Run co-occurrence-based learning (using torch.nn.Modul)
        :param Xd: Disretized data (represented as cluster indexes)
        :param Xc: Original, continuous (gaussian) data
        :param lengths: Lengths of individual sequences in X
        """
        # TODO: iterate
        # TODO: loguj do convergence monitora raz na jakiś czas
        # TODO: parametrize

        self.model.to(device)
        cooc_matrix = torch.tensor(self._cooccurence(Xd, lengthsd)).to(device)
        run = self.optim_params.pop('run') if 'run' in  self.optim_params.keys() else None
        optimizer = self.optimizer(self.model.parameters(), **self.optim_params)
        nodes_tensor = torch.Tensor(self.nodes.T).to(device)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        for i in range(self.max_epoch):
            optimizer.zero_grad()
            loss = torch.nn.KLDivLoss(reduction="sum")(
                torch.log(self.model(nodes_tensor)), cooc_matrix
            )
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                (
                    self.NFs,
                    self.transmat_,
                    self.startprob_,
                ) = self.model.get_model_params()
                self.emissionprob_ = torch.nn.functional.normalize(
                    torch.cat(
                        [
                            torch.exp(self.model.emission_score(dist, nodes_tensor)).reshape(
                                1, -1
                            )
                            for dist in self.NFs
                        ],
                        dim=0,
                    ),
                    dim=1, p=1
                ).cpu().detach().numpy()

                if run is not None:
                    run.log({"score": self.score(Xc, lengthsc), "loss": loss.cpu().detach()})

            elif i % 1000 == 999:  # TODO: select properly
                (
                    self.NFs,
                    self.transmat_,
                    self.startprob_,
                ) = self.model.get_model_params()

                scheduler.step()
                self.emissionprob_ = torch.nn.functional.normalize(
                    torch.cat(
                        [
                            torch.exp(self.model.emission_score(dist, nodes_tensor)).reshape(
                                1, -1
                            )
                            for dist in self.NFs
                        ],
                        dim=0,
                    ),
                    dim=1, p=1
                ).cpu().detach().numpy()
                if Xc is not None:
                    score = self.score(Xd.reshape(-1, 1), lengthsd)
                    self.monitor_.report(score)
                    if (
                        early_stopping and
                        self.monitor_.converged
                    ):  # TODO: monitor convergence from torch training
                        break
        (
            self.NFs,
            self.transmat_,
            self.startprob_,
        ) = self.model.get_model_params()

    def fit(
        self,
        X: npt.NDArray,
        lengths: Optional[
            npt.NDArray[int]
        ] = None,  # we can possible specify other data for training and validation
        Xd: Optional[npt.NDArray] = None,
        lengths_d: Optional[npt.NDArray[int]] = None,
        update_nodes: bool = False,

        early_stopping: bool = False
    ):
        # TODO: fix docstrings
        """
        Train the model tih the proper method
        :param X: Original, continuous (gaussian) data
        :param lengths_d: Lengths of individual sequences in X
        :param update_nodes: Should the nodes be re-initialized, if they are already provided.
        :return:
        """

        if Xd is None:
            Xd = self.discretize(X, update_nodes)
            lengths_d = lengths
        super()._init(Xd)
        self._init(X, lengths)
        if self.learning_alg == "em":
            super().fit(self.nodes.T[Xd], lengths_d)
        elif self.learning_alg == "em_dense":
            self._fit_em_dense(X, lengths_d)
        elif self.learning_alg == "cooc":
            self._fit_cooc(Xd, X, lengths_d, lengths, early_stopping)
        else:
            _log.error(
                f"Learning algorithm {self.learning_alg} is not implemented. Select one of: {LEARNING_ALGORITHMS}"
            )

    def score(self, X, lengths):
        Xd = self.discretize(X, force=False).reshape(-1, 1)
        return super().score(Xd, lengths)

    def predict(self, X, lengths=None):
        Xd = self.discretize(X, force=False).reshape(-1, 1)
        return super().predict(Xd, lengths)



if __name__ == "__main__":
    hmm = hmm.GaussianHMM(3).fit(np.random.normal(0, 1, 100).reshape(-1, 1))
    obs, hid = hmm.sample(100)

    myHMM = FlowHMM("random", 10, n_components=3, learning_alg="cooc", verbose=True)
    myHMM2 = FlowHMM(
        "uniform", 10, n_components=3, learning_alg="cooc", verbose=True
    )
    myHMM3 = FlowHMM(
        "latin_cube_u", 10, n_components=3, learning_alg="cooc", verbose=True
    )
    myHMM4 = FlowHMM(
        "latin_cube_q", 10, n_components=3, learning_alg="cooc", verbose=True
    )

    myHMM5 = FlowHMM(
        "sobol", 10, n_components=3, learning_alg="cooc", verbose=True
    )

    myHMM6 = FlowHMM(
        "halton", 10, n_components=3, learning_alg="cooc", verbose=True
    )

    myHMM.fit(obs)
    myHMM2.fit(obs)
    myHMM3.fit(obs)
    myHMM4.fit(obs)
    myHMM5.fit(obs)
    myHMM6.fit(obs)
