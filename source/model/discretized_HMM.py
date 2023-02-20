from hmmlearn import hmm
import numpy as np


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
        self.nodes = number_of_nodes
        # Placeholders
        self.nodes = None

    def _provide_nodes_random(self):
        pass

    def _provide_nodes_latin(self):
        pass

    def _provide_nodes_uniform(self):
        pass

    def _provide_nodes(self, X, force):
        if not force and (self.nodes is not None):
            print("Nodes had been set previously. Use force=True to update them")
            pass
        elif self.discretization_method == 'random':
            self._provide_nodes_random()
        elif self.discretization_method == 'latin_cube':
            self._provide_nodes_latin()
        else:
            self._provide_nodes_uniform()
        self.nodes = self.nodes.reshape(1, -1)

    def _discretize(self, X, force):
        self._provide_nodes(X, force)
        np.argmin((X - self.nodes).abs(), axis=1)
        pass

    def fit(self, X, lengths=None, force=False):
        pass

    def _do_mstep(self, stats):
        pass


if __name__ == "__main__":
    myHMM = DiscreteHMM()
