from hmmlearn import hmm
from hmmlearn.base import _log
import numpy as np
from scipy.stats.qmc import LatinHypercube
# import tensorflow as tf
import torch

# TODO: add possible embeddings (?)

DISCRETIZATION_TECHNIQUES = ['random', 'latin_cube_u', 'latin_cube_q', 'uniform']
OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)
LEARNING_ALGORITHMS = ['em_dense', 'em_discrete', 'cooc']

# TODO: try reading data out of file when using quasi random nodes


class DiscreteHMM(hmm.GaussianHMM):
    def __init__(self,
                 discretization_method: str = 'random', number_of_nodes: int = 100, embedding_dim=10,
                 n_components=1, startprob_prior=1.0, transmat_prior=1.0,
                 covariance_type='diag', min_covar=0.001,
                 means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
                 algorithm='viterbi', random_state=None, n_iter=10, tol=0.01, verbose=False, params='stmc',  # TODO: default without 's'
                 init_params='stmc', implementation='log') -> None:

        super(DiscreteHMM, self).__init__(
            n_components=n_components, covariance_type=covariance_type, min_covar=min_covar,
            startprob_prior=startprob_prior, transmat_prior=transmat_prior,
            means_prior=means_prior, means_weight=means_weight, covars_prior=covars_prior, covars_weight=covars_weight,
            algorithm=algorithm, random_state=random_state, n_iter=n_iter, tol=tol, verbose=verbose,
            params=params, init_params=init_params, implementation=implementation)

        assert discretization_method in DISCRETIZATION_TECHNIQUES, \
            f"discretization method: '{discretization_method}' not allowed, choose one of {DISCRETIZATION_TECHNIQUES}"

        self.discretization_method = discretization_method
        self.no_nodes = number_of_nodes
        self.nodes = None  # Placeholder
        self.optimizer = None  # TODO
        self.l = embedding_dim

    def _provide_nodes_random(self, X):
        self.nodes = X[np.random.choice(X.shape[0], size=self.no_nodes, replace=False)].transpose()

    def _provide_nodes_latin_q(self, X):  # each point in a row
        self.nodes = np.apply_along_axis(
            lambda x: np.quantile(x[:(-self.no_nodes)], x[(-self.no_nodes):]), 0,
            np.concatenate([X, LatinHypercube(self.no_nodes).random(X.shape[1]).transpose()], axis=0)).transpose()

    def _provide_nodes_latin_u(self, X):  # each point in a row
        self.nodes = (LatinHypercube(self.no_nodes).random(X.shape[1]).transpose() *
                      (X.max(axis=0) - X.min(axis=0))[np.newaxis, :] + X.min(axis=0)[np.newaxis, :]).transpose()

    def _provide_nodes_uniform(self, X):
        self.nodes = (np.random.uniform(size=self.no_nodes * X.shape[1]).reshape(self.no_nodes, X.shape[1]) *
                      (X.max(axis=0) - X.min(axis=0))[np.newaxis, :] + X.min(axis=0)[np.newaxis, :]).transpose()

    def _provide_nodes(self, X, force):
        if not force and (self.nodes is not None):
            if self.verbose:
                print("Nodes have been already set. Use force=True to update them")
            pass
        elif self.discretization_method == 'random':
            self._provide_nodes_random(X)
        elif self.discretization_method == 'latin_cube_q':
            self._provide_nodes_latin_q(X)
        elif self.discretization_method == 'latin_cube_u':
            self._provide_nodes_latin_u(X)
        else:
            self._provide_nodes_uniform(X)

    def _discretize(self, X, force):
        self._provide_nodes(X, force)
        return np.argmin(np.square(X[:, :, np.newaxis] - self.nodes[np.newaxis, :, :]).sum(axis=1), axis=1).reshape(-1)

    def _needs_init(self, code, name, recheck=True):
        result = True if code in self.init_params else False

        if code == "t":
            if "s" in self.init_params:
                _log.warning("Optimizing separately attribute 'startprob_' ignores the stationarity requirement")
        if code in ["t", "s"]:
            if "z" in self.init_params and "u" in self.init_params:
                _log.warning("Attributes 'startprob_' and 'transmat_' will be initialized based on "
                             "attributes 'u_' and 'z_'")
                result = False
        if code in ["z", "u"]:
            result = (code in self.params) and (code in self.init_params)
        return result

    def _init(self, X, lengths=None):
        super()._init(X)
        if self._needs_init("m", "means_"):
            self._means_tensor = torch.tensor(self.means_, requires_grad=True)
        if self._needs_init("c", "covars_"):
            self._covars_tensor = torch.tensor(self.covars_, requires_grad=True)

        if self._needs_init("z", "z_") and self._needs_init("u", "u_"):
            # self._z_tensor = torch.tensor(None, requires_grad=True)  # TODO: use embedding lentgh l
            # self._u_tensor = torch.tensor(None, requires_grad=True)
            if self._needs_init("s", "startprob_"):
                self._startprob_tensor = torch.tensor(self.startprob_,
                                                      requires_grad=False)  # TODO: specify based on z, u
            if self._needs_init("t", "transmat_"):
                self._transmat_tensor = torch.tensor(self.transmat_, requires_grad=False)
        else:
            if self._needs_init("s", "startprob_"):
                self._startprob_tensor = torch.tensor(self.startprob_, requires_grad=True)
            if self._needs_init("t", "transmat_"):
                self._transmat_tensor = torch.tensor(self.transmat_, requires_grad=True)

    def fit(self, X, lengths=None, update_nodes=False, learning_alg='em_dense'):
        Xd = self._discretize(X, update_nodes)
        if learning_alg == 'em_dense':
            super().fit(self.nodes.T[Xd], lengths)  # TODO
        elif learning_alg == 'em_discrete':
            pass
        elif learning_alg == 'cooc':
            pass
        else:
            _log.error(f'Learning algorithm {learning_alg} is not implemented. Select one of: {LEARNING_ALGORITHMS}')
    def _do_mstep(self, stats):  # TODO
        super()._do_mstep(stats)


if __name__ == "__main__":
    hmm = hmm.GaussianHMM(3).fit(np.random.normal(0, 1, 100).reshape(-1, 1))
    myHMM = DiscreteHMM('random', 10, 3)
    myHMM2 = DiscreteHMM('uniform', 10, 3)
    myHMM3 = DiscreteHMM('latin_cube_u', 10, 3)
    myHMM4 = DiscreteHMM('latin_cube_q', 10, 3)
    obs, hid = hmm.sample(100)
    myHMM.fit(obs)
    myHMM2.fit(obs)
    myHMM3.fit(obs)
    myHMM4.fit(obs)
