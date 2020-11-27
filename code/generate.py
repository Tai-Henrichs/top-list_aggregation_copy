import mallows_kendall as mk
import numpy as np

def generateSample(m,n,k_distribution,theta=None, phi=None, s0=None):
    """This function generates a single sample according
    to Mallows Models adapted to top-k rankings given a parameter of dispersion
    (theta or phi), where the values of k are controlled by k_distribution
        Parameters
        ----------
        m: int
            The number of rankings to generate
        n: int
            The number of candidates considered by rankers. Note that
            k <= n, since rankers cannot create preference lists that
            rank more candidates than exist.
        theta: float, optional (if phi given)
            The dispersion parameter theta
        phi: float, optional (if theta given)
            The dispersion parameter phi
        k_distribution: dict [int : float or double]
            The provided dictionary indicates the quantity of
            lists in the returned sample of each length. A given
            [key : value] entry dictates that (value * m) top-key-lists will
            be created in the resulting sample. Uses default Python rounding.
        s0: ndarray
            The consensus ranking. The identity ranking by default.
        Returns
        -------
        list
            The top-lists generated
    """

    return np.array([rank[:k] for
        rank in mk.sampling_top_k_rankings(m * prop, n, k, theta, phi, s0) for
        k, prop in k_distribution])

def generateSamples(m,n,k_distribution,thetas=None, phis=None, s0=None):
    """Generates one sample for each parameter of dispersion given, i.e.
        for each theta or for each phi
        Parameters
        ----------
        m: int
            The number of rankings to generate
        n: int
            The number of candidates considered by rankers. Note that
            k <= n, since rankers cannot create preference lists that
            rank more candidates than exist.
        theta: list of floats, optional (if phi given)
            The dispersion parameter theta
        phi: list of floats, optional (if theta given)
            The dispersion parameter phi
        k_distribution: dict [int : float or double]
            The provided dictionary indicates the quantity of
            lists in the returned sample of each length. A given
            [key : value] entry dictates that (value * m) top-key-lists will
            be created in the resulting sample. Uses default Python rounding.
        s0: ndarray
            The consensus ranking. The identity ranking by default.
        Returns
        -------
        list
            List of samples generated
    """
    return [generateSample(m,n,k_distribution, theta, phi, s0) for
            theta, phi in zip(thetas, phis)]


    # Main function generates synthetic data-sets using the Mallows-Model
    # and saves the output in CSV format.
    # Two files are generated per-sample, one treating duplicate top-lists as
    # separate entries and the other outputting only unique top-lists with
    # frequency information accompanying each top-list.
    # Primarily for testing purposes.
    if __name__ == '__main__':
