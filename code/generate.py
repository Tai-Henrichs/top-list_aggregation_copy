import mallows_kendall as mk
import numpy as np
import collections
from collections import Counter

class MallowsSample:
    def topListSample(self,n,k_distribution,theta=None,phi=None,s0=None):
        """This function generates a single sample according
        to Mallows Models adapted to top-k rankings given a parameter of dispersion
        (theta or phi), where the values of k are controlled by k_distribution
            Parameters
            ----------
            n: int
                The number of candidates considered by rankers. Note that
                k <= n, since rankers cannot create preference lists that
                rank more candidates than exist.
            k_distribution: dict {int : int}
                The provided dictionary indicates the quantity of
                lists in the returned sample of each length. A given
                [key : value] entry dictates that value top-key-lists will
                be created in the resulting sample.
            theta: float, optional (if phi given)
                The dispersion parameter theta
            phi: float, optional (if theta given)
                The dispersion parameter phi
            s0: ndarray
                The consensus ranking. The identity ranking by default.
            Returns
            -------
            list
                The top-lists generated
        """
        return [tuple(ranking[~np.isnan(ranking)]) for 
                k, freq in k_distribution.items() for 
                    ranking in mk.sampling_top_k_rankings(freq, n, k, theta, phi, s0)]

    """This class represents a single sample generated from a
    Mallows Models adapted to top-k rankings given a parameter of dispersion
    (theta or phi), and where the lengths and quantities of the top-lists are
    dictated by k_distribution.
        Parameters
        ----------
        n: int
            The number of candidates considered by rankers. Note that
            k <= n, since rankers cannot create preference lists that
            rank more candidates than exist.
        k_distribution: dict {int : int}
            The provided dictionary indicates the quantity of
            lists in the returned sample of each length. A given
            [key : value] entry dictates that value top-key-lists will
            be created in the resulting sample.
        theta: float, optional (if phi given)
            The dispersion parameter theta
        phi: float, optional (if theta given)
            The dispersion parameter phi
        s0: ndarray
            The consensus ranking. The identity ranking by default.
    """
    def __init__(self,n,k_distribution,theta=None,phi=None,s0=None):
        self.n = n
        self.theta, self.phi = mk.check_theta_phi(theta, phi)
        self.s0 = np.array(range(n)) if s0 is None else s0
        self.sample = self.topListSample(n,k_distribution,theta,phi,s0)
        self.m = len(self.sample)
        self.sample = Counter(self.sample)

    sampleType = "Mallows"

    def label(self):
        precision = 2
        return (f"{self.sampleType}_"
                f"candidates-{self.n}_"
                f"voters-{self.m}_"
                f"theta-{self.theta:.{precision}f}_"
                f"phi-{self.phi:.{precision}f}"
                )

    def __str__(self):
        return (self.label() + "\n"
                f"Rankings:\n{self.sample}\n"
                "Consensus Ranking: " + np.array_str(self.s0) + "\n"
                )

class MallowsSampleTopK(MallowsSample):
    sampleType = "Mallows_Top-K"

    """This class represents a sample of m top-k-lists sampled from a
    Mallows Models adapted to top-k rankings given a parameter of dispersion
    (theta or phi).
    Parameters
    ----------
    n: int
    The number of candidates considered by rankers. Note that
    k <= n, since rankers cannot create preference lists that
    rank more candidates than exist.
    m: int
    The number of rankings to generate.
    theta: float, optional (if phi given)
    The dispersion parameter theta
    phi: float, optional (if theta given)
    The dispersion parameter phi
    k: int
    number of known positions of items for the rankings.
    s0: ndarray
    The consensus ranking. The identity ranking by default.
    Returns
    -------
    ndarray
    The top-lists generated
    """
    def __init__(self,m,n,k,theta=None,phi=None,s0=None):
        self.m = m
        self.k = k
        k_distribution = {k : m}
        super().__init__(n,k_distribution,theta,phi,s0)

    def label(self):
        return super().label() + f"_k-{self.k}"

class MallowsSamplePoisson(MallowsSample):
    sampleType = "Mallows_Poisson"

    def poissonSample(self,m,lda,lower = 0,upper = float('inf')):
        """Returns m values drawn from a Poisson distribution on lambda for
            which all values are in [lower, upper] through repeated sampling.
            Not guaranteed to terminate.
            Parameters
            ----------
            m: int
                The number of elements in the requested sample
            lda: float
                The lambda parameter on which the Poisson distribution is
                defined.
            lower: float
                Lowerbound of the sampled values
            upper: float
                Upperbound of the sampled values.
        """
        rng = np.random.default_rng()
        # Samples are drawn repeatedly until enough values
        # reside in [lower,upper]
        sampleSize = m
        while True:
            sample = rng.poisson(lda, sampleSize)
            sample = sample[(sample <= upper) & (sample >= lower)]
            if len(sample) >= m:
                return sample[:m]
            # Double the number of values sampled each time to
            # increase the chance that sufficient values lay in [lower,upper]
            sampleSize *= 2

    """This class represents a single sample generated from
        a Mallows Models adapted to top-k rankings given a parameter of dispersion
        (theta or phi). The lengths of rankers' top-lists are sampled from a
        Poisson distribution on lda. With m top-lists, we define one m_i for
        each i in [1,m], where each m_i is randomly sampled from a truncated
        Poisson distribution with the interval [1,n].
        Then, we have a top-m_i-list in the sample for
        each m_i, for m top-lists in total.

        Parameters
        ----------
        m: int
            The number of rankings to generate.
        n: int
            The number of candidates considered by rankers.
        lda: float
            The parameter on which the Poisson distribution is defined, short
            for lambda.
            If lda is an integer, the median number of candidates
            ranked by voters will be lda. lda is always the expected value of
            an element randomly sampled from a Poisson distribution on lda.
        theta: list of floats, optional (if phi given)
            The dispersion parameter theta
        phi: list of floats, optional (if theta given)
            The dispersion parameter phi
        s0: ndarray
            The consensus ranking. The identity ranking by default.
    """
    def __init__(self,m,n,lda,theta=None,phi=None,s0=None):
        self.m = m
        self.lda = lda
        k_distribution = Counter(self.poissonSample(m,lda,1,n))
        super().__init__(n,k_distribution,theta,phi,s0)

    def label(self):
        return super().label() + f"_lambda-{self.lda}"

# Main function generates synthetic data-sets using the Mallows-Model
# and saves the output in CSV format. The data-sets are made for the
# the following dispersion parameters: theta = .01, theta = .1
# For each value of theta, two samples are generated, one where all
# lists rank the top 5 of 10 candidates, and another where top-lists have
# their lengths sampled from a Poisson distribution.
# Note: Primarily for testing and debugging.
if __name__ == '__main__':
    # General Sample parameters
    m = 20
    n = 10
    thetas = (.01, .1)

    k = 5
    lda = 2
    for theta in thetas:
         # Samples of top-5-lists
        print(MallowsSampleTopK(m,n,k,theta))

        # Samples of top-lists with lengths sampled from Poisson distribution
        print(MallowsSamplePoisson(m,n,lda,theta))
