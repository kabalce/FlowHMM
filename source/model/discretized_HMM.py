from hmmlearn import hmm
from hmmlearn.base import _log
import numpy as np
import numpy.typing as npt
from scipy.stats.qmc import LatinHypercube
import torch

# czy jest hmm w torchu już zaimplementowany:
# http://torch.ch/torch3/manual/HMM.html
# https://github.com/nwams/Hidden_Markov_Model-in-PyTorch
# could be useful also: https://pyro4ci.readthedocs.io/en/latest/_modules/pyro/distributions/hmm.html

# hints for implementation: TYPOWANIE!

# TODO: recheck cavariance learning in HmmOptim!
# TODO: recheck Latin cube


# Future features:
# try reading data out of file when using quasi random nodes
# torch model with embeddings in separate class


DISCRETIZATION_TECHNIQUES = ["random", "latin_cube_u", "latin_cube_q", "uniform"]
OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)
LEARNING_ALGORITHMS = ["em", "em_dense", "cooc"]


class HmmOptim(torch.nn.Module):
    def __init__(
        self,
        n_components,
        n_dim,
        means_=None,
        covars_=None,
        startprob_=None,
        transmat_=None,  # Initial values
        trainable: str = "",
        trans_from="S",
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
        super(HmmOptim, self).__init__()

        # TODO: implement various covariance types (now works only for full)

        means = (
            means_
            if means_ is not None
            else np.random.standard_normal(n_components * n_dim).reshape(
                n_components, n_dim
            )
        )

        covar_L = (
            np.linalg.cholesky(covars_)
            if covars_ is not None
            else np.tril(
                np.random.standard_normal(n_components * n_dim**2).reshape(
                    (n_components, n_dim, n_dim)
                )
            )
        )

        transmat = (
            transmat_
            if transmat_ is not None
            else np.random.standard_normal(n_components * n_components).reshape(
                n_components, n_components
            )
        )
        transmat /= transmat.sum(axis=1)[:, np.newaxis]

        startprob = (
            startprob_
            if startprob_ is not None
            else np.random.standard_normal(n_components)
        )
        startprob /= startprob.sum()

        self.n_components = n_components
        self.n_dim = n_dim
        self.trainable = trainable
        self._means_tensor = torch.nn.Parameter(
            torch.tensor(means), requires_grad="m" in trainable
        )
        self._covar_L_tensor = torch.nn.Parameter(
            torch.tensor(covar_L), requires_grad="c" in trainable
        )
        self._S_unconstrained = torch.nn.Parameter(
            torch.tensor(np.log(transmat * startprob)), requires_grad="t" in trainable
        )

    def forward(self, nodes: npt.NDArray):
        """
        Calculate the forward pass of the torch.nn.Module
        :return: cooc matrix from current parameters
        """
        covars = self._covar_L_tensor @ torch.transpose(self._covar_L_tensor, 1, 2)
        distributions = [
            torch.distributions.MultivariateNormal(self._means_tensor[i], covars[i])
            for i in range(self.n_components)
        ]

        B = torch.nn.functional.normalize(
            torch.cat(
                [
                    dist.log_prob(torch.Tensor(nodes.T)).reshape(1, -1)
                    for dist in distributions
                ],
                dim=0,
            ),
            dim=1,
        )

        Ss = torch.exp(self._S_unconstrained)
        S = Ss / Ss.sum()
        return B.T @ S @ B

    @staticmethod
    def _to_numpy(tens):
        """
        Get value of torch tensor as a numpy array
        :param tens: torch tensor (or parameter)
        :return: numpy array
        """
        return tens.detach().numpy()  # TODO: check if it will be working in all cases

    def get_model_params(self):
        """
        Retrieve HMM parameters from torch.nn.Module
        :return: means, covars, transmat, startprob
        """
        # TODO: https://github.com/tooploox/flowhmm/blob/main/src/flowhmm/models/fhmm.py linijka 336 - czy mi to potrzebne

        Ss = torch.exp(self._S_unconstrained)
        S = Ss / Ss.sum()
        startprob = torch.sum(S, dim=1)
        transmat = S / startprob.unsqueeze(1)

        covars = self._covar_L_tensor @ torch.transpose(self._covar_L_tensor, 1, 2)
        means = self._means_tensor
        return (
            self._to_numpy(means),
            self._to_numpy(covars),
            self._to_numpy(transmat),
            self._to_numpy(startprob),
        )


class DiscreteHMM(hmm.GaussianHMM):
    def __init__(
        self,
        discretization_method: str = "random",
        no_nodes: int = 100,
        l=None,
        learning_alg="cooc",
        n_components=1,
        startprob_prior=1.0,
        transmat_prior=1.0,
        optim_params=None,
        optimizer="SGD",
        covariance_type="full",
        min_covar=0.001,  # TODO: implement different covariance types
        means_prior=0,
        means_weight=0,
        covars_prior=0.01,
        covars_weight=1,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=0.01,
        verbose=False,
        params="tmc",  # TODO: default without 's'
        init_params="tmc",
        implementation="log",
    ) -> None:
        super(DiscreteHMM, self).__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            min_covar=min_covar,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            means_prior=means_prior,
            means_weight=means_weight,
            covars_prior=covars_prior,
            covars_weight=covars_weight,
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
            discretization_method in DISCRETIZATION_TECHNIQUES
        ), f"discretization method: '{discretization_method}' not allowed, choose one of {DISCRETIZATION_TECHNIQUES}"

        self.discretization_method = discretization_method
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

    def _provide_nodes_random(self, X):
        """
        Select random observations as nodes for discretization; nodes are saved in attribute nodes
        :param X: Original, continuous (gaussian) data
        """
        self.nodes = X[
            np.random.choice(X.shape[0], size=self.no_nodes, replace=False)
        ].transpose()

    def _provide_nodes_latin_q(self, X):
        """
        Provide nodes from CDF on latin qube; nodes are saved in attribute nodes
        :param X: Original, continuous (gaussian) data
        """
        self.nodes = np.apply_along_axis(
            lambda x: np.quantile(x[: (-self.no_nodes)], x[(-self.no_nodes) :]),
            0,
            np.concatenate(
                [X, LatinHypercube(self.no_nodes).random(X.shape[1]).transpose()],
                axis=0,
            ),
        ).transpose()

    def _provide_nodes_latin_u(self, X):  # each point in a row
        """
        Provide nodes from a latin qube on cuboid of observations; nodes are saved in attribute nodes
        :param X:  Original, continuous (gaussian) data
        """
        self.nodes = (
            LatinHypercube(self.no_nodes).random(X.shape[1]).transpose()
            * (X.max(axis=0) - X.min(axis=0))[np.newaxis, :]
            + X.min(axis=0)[np.newaxis, :]
        ).transpose()

    def _provide_nodes_uniform(self, X):
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

    def provide_nodes(self, X, force):
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
        elif self.discretization_method == "latin_cube_q":
            self._provide_nodes_latin_q(X)
        elif self.discretization_method == "latin_cube_u":
            self._provide_nodes_latin_u(X)
        else:
            self._provide_nodes_uniform(X)

    def _discretize(self, X, force):
        """
        Provide nodes for discretization and represent continuous data as cluster indexes
        :param X: Original, continuous (gaussian) data
        :param force: Should nodes be updated, if they are already provided.
        :return: Discretized data (index of cluster)
        """
        self.provide_nodes(X, force)
        return np.argmin(
            np.square(X[:, :, np.newaxis] - self.nodes[np.newaxis, :, :]).sum(axis=1),
            axis=1,
        ).reshape(-1)

    def _needs_init(self, code: str, name, torch_check=False):
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

    def _init(self, X, lengths=None):
        """
        Initialize model parameters and prepare data before training
        :param X: Original, continuous (gaussian) data
        :param lengths: Lengths of individual sequences in X
        """
        super()._init(X)  # init k-means with a batch of data (of some maximum size)?

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

        if self._needs_init("m", "means_", True):
            torch_inits["means_"] = self.means_
        if self._needs_init("c", "covars_", True):
            torch_inits["covars_"] = self.covars_
        if self._needs_init("z", "z_", True) and self._needs_init("u", "u_", True):
            torch_inits["z_"] = self.z_
            torch_inits["u_"] = self.u_
        elif self._needs_init("t", "transmat_", True):
            torch_inits["startprob_"] = self.compute_stationary(self.transmat_)
            torch_inits["transmat_"] = self.transmat_

        if self.learning_alg == "cooc":
            self.model = HmmOptim(**torch_inits)

    def _fit_em_dense(self, X, lengths=None):  # TODO: add for Gaussian Dense HMMs
        """
        TODO
        :param X:
        :param lengths:
        :return:
        """
        pass

    def _cooccurence(self, Xd, lengths=None):
        """
        Process discrete data sequences into co-occurrence matrix
        :param Xd: Disretized data (represented as cluster indexes)
        :param lengths: Lengths of individual sequences in X
        :return: co-occurrence matrix
        """
        # TODO: https://github.com/tooploox/flowhmm/blob/main/src/flowhmm/models/fhmm.py linijka 88 - skąd taki dzielnik??
        cooc_matrix = np.zeros(shape=(Xd.max() + 1, Xd.max() + 1))
        cont_seq_ind = np.ones(shape=Xd.shape[0])
        if lengths is None:
            lengths = np.array([Xd.shape[0]])
        cont_seq_ind[lengths.cumsum() - 1] *= 0
        for i in range(Xd.shape[0] - 1):
            cooc_matrix[Xd[i], Xd[i + 1]] += cont_seq_ind[i]
        cooc_matrix /= cooc_matrix.sum()
        return cooc_matrix

    def _fit_cooc(self, Xd, Xc, lengths=None):
        """
        Run co-occurrence-based learning (using torch.nn.Modul)
        :param Xd: Disretized data (represented as cluster indexes)
        :param Xc: Original, continuous (gaussian) data
        :param lengths: Lengths of individual sequences in X
        """
        # TODO: iterate
        # TODO: loguj do convergence monitora raz na jakiś czas
        # TODO: parametrize
        cooc_matrix = torch.tensor(self._cooccurence(Xd, lengths))
        optimizer = self.optimizer(self.model.parameters(), **self.optim_params)
        for i in range(self.max_epoch):
            optimizer.zero_grad()
            torch.nn.KLDivLoss(reduction="sum")(
                self.model(self.nodes), cooc_matrix
            ).backward()
            optimizer.step()
            if i % 1000 == 0:  # TODO: select properly
                (
                    self.means_,
                    self.covars_,
                    self.transmat_,
                    self.startprob_,
                ) = self.model.get_model_params()
                score = self.score(Xc, lengths)
                self.monitor_.report(score)
                if self.monitor_.converged:
                    break

    def fit(self, X, lengths=None, update_nodes=False):
        """
        Train the model tih the proper method
        :param X: Original, continuous (gaussian) data
        :param lengths: Lengths of individual sequences in X
        :param update_nodes: Should the nodes be re-initialized, if they are already provided.
        :return:
        """
        self._init(X, lengths)
        Xd = self._discretize(X, update_nodes)
        if self.learning_alg == "em":
            super().fit(self.nodes.T[Xd], lengths)
        elif self.learning_alg == "em_dense":
            self._fit_em_dense(X, lengths)
        elif self.learning_alg == "cooc":
            self._fit_cooc(Xd, X, lengths)
        else:
            _log.error(
                f"Learning algorithm {self.learning_alg} is not implemented. Select one of: {LEARNING_ALGORITHMS}"
            )


if __name__ == "__main__":


    hmm = hmm.GaussianHMM(3).fit(np.random.normal(0, 1, 100).reshape(-1, 1))
    obs, hid = hmm.sample(100)

    myHMM = DiscreteHMM("random", 10, n_components=3, learning_alg="cooc", verbose=True)
    myHMM2 = DiscreteHMM(
        "uniform", 10, n_components=3, learning_alg="cooc", verbose=True
    )
    myHMM3 = DiscreteHMM(
        "latin_cube_u", 10, n_components=3, learning_alg="cooc", verbose=True
    )
    myHMM4 = DiscreteHMM(
        "latin_cube_q", 10, n_components=3, learning_alg="cooc", verbose=True
    )

    myHMM.fit(obs)
    myHMM2.fit(obs)
    myHMM3.fit(obs)
    myHMM4.fit(obs)

# softmax - mozna jeszcze optymalizować mnożnik w wykłądniku - sprawdź różne lambdy
# twierdzenie że cooc działa przy dostatecznej liczbie danych (dla konkretnych liczb stanów)
# nmf - jednoznaczność

# D. Wegner, B. Chmiela - zobacz prace
