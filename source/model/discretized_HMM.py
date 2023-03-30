from hmmlearn import hmm
from hmmlearn.base import _log
import numpy as np
import numpy.typing as npt
from scipy.stats.qmc import LatinHypercube
# import tensorflow as tf
import torch

# czy jest hmm w torchu już zaimplementowany
# TYPOWANIE, DOCSTRINGI
# TODO: add possible embeddings (?)

DISCRETIZATION_TECHNIQUES = ['random', 'latin_cube_u', 'latin_cube_q', 'uniform']
OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)
LEARNING_ALGORITHMS = ['em', 'em_dense', 'cooc']

# TODO: try reading data out of file when using quasi random nodes


class HmmOptim(torch.nn.Module):
    def __init__(self, means_=None, covars_=None, startprob_=None, transmat_=None,  # Initial values
                 trainable: str = ""):
        """
        TODO
        :param cooc_matrix:
        :param nodes:
        :param means_:
        :param covars_:
        :param startprob_:
        :param transmat_:
        :param trainable:
        """
        super().__init__()
        # TODO: make customizable and possible dependent on embeddings
        # TODO: provide initial values!
        # TODO: init parameters with specified values or random values (if None)
        self.trainable = trainable
        self._means_tensor = torch.nn.Parameter(means_, requires_grad=True)  # PARAMETERS, propaguj
        self._covars_tensor = torch.nn.Parameter(covars_, requires_grad=True)
        # self._startprob_tensor = torch.nn.Parameter(startprob_, requires_grad=False)
        # self._transmat_tensor = torch.nn.Parameter(transmat_, requires_grad=False)
        self._S_unconstrained = torch.nn.Parameter(np.log(transmat_ * startprob_), requires_grad=True)  # TODO: or embedding

    def _check_trainable(self, code):
        return code in self.trainable

    def forward(self, nodes: npt.NDArray):
        """
        TODO
        :return:
        """
        distributions = [torch.distributions.MultivariateNormal(torch.tensor(self.means_[i]),
                                                                torch.tensor(self.covars_[i])) for i in
                         range(self.means_.shape[0])]

        B = torch.nn.functional.normalize(
            torch.cat(
                [dist.log_prob(torch.Tensor(nodes.T)).reshape(1, -1) for dist in distributions],
                dim=0),
            dim=1)

        S = torch.softmax(self._S_unconstrained, dim=1)
        return B.T @ S @ B


    def _get_transmat(self):
        """
        TODO: compute standard transition matrix from self._S_unconstrained
        :return:
        """
        pass


class DiscreteHMM(hmm.GaussianHMM):
    def __init__(self,
                 discretization_method: str = 'random', no_nodes: int = 100, l=None, learning_alg='em_dense',
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
        self.no_nodes = no_nodes
        self.nodes = None  # Placeholder
        self.optimizer = None  # TODO
        self.model = None
        self.learning_alg = learning_alg
        # TODO: make it optional
        self.l = l
        self.z_, self.u_ = None, None

    def _provide_nodes_random(self, X):
        """
        TODO
        :param X:
        :return:
        """
        self.nodes = X[np.random.choice(X.shape[0], size=self.no_nodes, replace=False)].transpose()

    def _provide_nodes_latin_q(self, X):  # each point in a row
        """
        TODO
        :param X:
        :return:
        """
        self.nodes = np.apply_along_axis(
            lambda x: np.quantile(x[:(-self.no_nodes)], x[(-self.no_nodes):]), 0,
            np.concatenate([X, LatinHypercube(self.no_nodes).random(X.shape[1]).transpose()], axis=0)).transpose()

    def _provide_nodes_latin_u(self, X):  # each point in a row
        """
        TODO
        :param X:
        :return:
        """
        self.nodes = (LatinHypercube(self.no_nodes).random(X.shape[1]).transpose() *
                      (X.max(axis=0) - X.min(axis=0))[np.newaxis, :] + X.min(axis=0)[np.newaxis, :]).transpose()

    def _provide_nodes_uniform(self, X):
        """
        TODO
        :param X:
        :return:
        """
        self.nodes = (np.random.uniform(size=self.no_nodes * X.shape[1]).reshape(self.no_nodes, X.shape[1]) *
                      (X.max(axis=0) - X.min(axis=0))[np.newaxis, :] + X.min(axis=0)[np.newaxis, :]).transpose()

    def _provide_nodes(self, X, force):
        """
        TODO
        :param X:
        :param force:
        :return:
        """
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
        """
        TODO
        :param X:
        :param force:
        :return:
        """
        self._provide_nodes(X, force)
        return np.argmin(np.square(X[:, :, np.newaxis] - self.nodes[np.newaxis, :, :]).sum(axis=1), axis=1).reshape(-1)

    def _needs_init(self, code, name, torch_check=False):
        """
        TODO
        :param code:
        :param name:
        :param torch_check:
        :return:
        """
        if torch_check:
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
        else:
            result = super()._needs_init(code, name)
        return result

    @staticmethod
    def compute_stationary(matrix):  # TODO: or maybe it should be a separate function -- think of it
        """
        TODO
        :param matrix:
        :return:
        """
        vals, vecs = np.linalg.eig(matrix.T)
        vec1 = vecs[:, np.isclose(vals, 1)].reshape(-1)
        stationary = vec1 / vec1.sum()
        return stationary.real

    def _init(self, X, lengths=None):
        """
        TODO
        :param X:
        :param lengths:
        :return:
        """
        super()._init(X)  # init k-means with a batch of data (of some maximum size)?

        for e in ['z', 'u']:
            if self._needs_init(e, f"{e}_"):
                setattr(self, f"{e}_", np.random.standard_normal(self.l * self.n_components).reshape(self.l, self.n_components))

        torch_inits = dict()

        if self._needs_init("m", "means_", True):
            torch_inits['means_'] = self.means_
        if self._needs_init("c", "covars_", True):
            torch_inits['covars_'] = self.covars_
        if self._needs_init("z", "z_", True) and self._needs_init("u", "u_", True):
            torch_inits['z_'] = self.z_
            torch_inits['u_'] = self.u_
        elif self._needs_init("t", "transmat_", True):
            torch_inits['startprob_'] = self.compute_stationary(self.transmat_)
            torch_inits['transmat_'] = self.startprob_

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
        TODO
        :param Xd:
        :param lengths:
        :return:
        """
        cooc_matrix = torch.tensor(np.zeros(shape=(Xd.max(), Xd.max())))
        cont_seq_ind = np.ones(shape=Xd.max())
        cont_seq_ind[lengths.cumsum() - 1] *= 0
        for i in range(Xd.shape[0] - 1):
            cooc_matrix[Xd[i], Xd[i+1]] += cont_seq_ind[i]
        cooc_matrix /= Xd.shape[0]
        return cooc_matrix

    def _fit_cooc(self, X, lengths=None):
        """
        TODO
        :param X:
        :param lengths:
        :return:
        """
        cooc_matrix = self._cooccurence(X, lengths)
        pass

    def fit(self, X, lengths=None, update_nodes=False):
        """
        TODO
        :param X:
        :param lengths:
        :param update_nodes:
        :return:
        """
        Xd = self._discretize(X, update_nodes)
        if self.learning_alg == 'em':
            super().fit(self.nodes.T[Xd], lengths)
        elif self.learning_alg == 'em_dense':
            self._fit_em_dense(X, lengths)
        elif self.learning_alg == 'cooc':
            self._fit_cooc(X, lengths)
        else:
            _log.error(f'Learning algorithm {self.learning_alg} is not implemented. Select one of: {LEARNING_ALGORITHMS}')


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

# softmax - mozna jeszcze optymalizować mnożnik w wykłądniku - sprawdź różne lambdy
# twierdzenie że cooc działa przy dostatecznej liczbie danych (dla konkretnych liczb stanów)
# nmf - jednoznaczność

# D. Wegner, B. Chmiela - zobacz prace