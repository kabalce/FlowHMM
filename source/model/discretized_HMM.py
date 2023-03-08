from hmmlearn import hmm
import numpy as np
from scipy.stats.qmc import LatinHypercube


DISCRETIZATION_TECHNIQUES = ['random', 'latin_cube', 'uniform']


class DiscreteHMM(hmm.CategoricalHMM):
    def __init__(self, discretization_method: str = 'random', number_of_nodes: int = 100,
                 n_components=1, startprob_prior=1.0, transmat_prior=1.0, *, emissionprob_prior=1.0,  # n_features=None,
                 algorithm='viterbi', random_state=None, n_iter=10, tol=0.01, verbose=False, params='ste',
                 init_params='ste', implementation='log') -> None:
        super().__init__(n_components, startprob_prior, transmat_prior,
                         emissionprob_prior=emissionprob_prior,  # n_features=n_features,
                         algorithm=algorithm, random_state=random_state, n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params, implementation=implementation)
        assert discretization_method in DISCRETIZATION_TECHNIQUES, \
            f"discretization method: '{discretization_method}' not allowed, choose one of {DISCRETIZATION_TECHNIQUES}"
        self.discretization_method = discretization_method
        self.no_nodes = number_of_nodes
        # Placeholders
        self.nodes = None

    def _provide_nodes_random(self, X):
        self.nodes = X[np.random.choice(X.shape[0], size=self.no_nodes, replace=False)]

    def _provide_nodes_latin(self, X):  # each point in a row
        self.nodes = np.apply_along_axis(lambda x: np.quantile(x[:(-self.no_nodes)], x[(-self.no_nodes):]), 0, np.concatenate([X, LatinHypercube(self.no_nodes).random(X.shape[1]).transpose()],  axis=0)).transpose()

    def _provide_nodes_uniform(self, X):
        self.nodes = np.random.uniform(size=10).reshape(-1, 1) @ (X.max(axis=0) - X.min(axis=0)).reshape(1, -1) + X.min(axis=0)

    def _provide_nodes(self, X, force):
        if not force and (self.nodes is not None):
            print("Nodes had been set previously. Use force=True to update them")
            pass
        elif self.discretization_method == 'random':
            self._provide_nodes_random(X)
        elif self.discretization_method == 'latin_cube':
            self._provide_nodes_latin(X)
        else:
            self._provide_nodes_uniform(X)
        self.nodes = self.nodes.reshape(1, -1)

    def _discretize(self, X, force):
        self._provide_nodes(X, force)
        return np.argmin(np.abs(X - self.nodes), axis=1).reshape(-1, 1)

    def fit(self, X, lengths=None, force=False):
        Xd = self._discretize(X, force)
        super().fit(Xd, lengths)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)


if __name__ == "__main__":  # TODO: provide a test for an simple example
    hmm = hmm.GaussianHMM(3).fit(np.random.normal(0, 1, 100).reshape(-1, 1))
    myHMM = DiscreteHMM('random', 10, 3)
    myHMM2 = DiscreteHMM('uniform', 10, 3)
    myHMM3 = DiscreteHMM('latin_cube', 10, 3)
    obs, hid = hmm.sample(100)
    myHMM.fit(obs)
    myHMM2.fit(obs)
    myHMM3.fit(obs)
    # TODO: visualize the nodes
