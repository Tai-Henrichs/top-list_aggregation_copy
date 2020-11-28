import mallows_kendall as mk
import numpy as np
import collections
from collections import Counter

class MallowsSample:
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
        k_distribution: dict [int : int]
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
    def ___init___(self,n,k_distribution,theta=None,phi=None,s0=None):
        self.n = n
        self.theta = theta
        self.phi = phi
        self.s0 = s0
        self.sample = topListSample(self,n,k_distribution,theta,phi,s0)

    sampleType = "Mallows"

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
            k_distribution: dict [int : int]
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
            ndarray
                The top-lists generated
        """
        return np.array([ ranking[ranking is not np.nan] for
            ranking in mk.sampling_top_k_rankings(freq, n, k, theta, phi, s0) for
            k, freq in k_distribution])

    def label(self):
        return f"{sampleType}_candidates-{n}_theta-{theta}_phi-{phi}"

    def __str__(self):
        return label(self) + "\n" + "Rankings: " + np.array_str(sample) + "\n" +
                "Consensus Ranking: " + np.array_str(s0)

class MallowsSampleTopK(MallowsSample):
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
    def ___init___(self,m,n,k,theta=None,phi=None,s0=None):
        self.m = m
        self.k = k
        super().___init___(n, {k : m},theta,phi,s0)

    def label(self):
        return super().label() + f"_K-{K}"

    sampleType = "Mallows_Top-K"

class MallowsSamplePoisson(MallowsSample):
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
    def ___init___(self,m,n,lda,theta=None,phi=None,s0=None):
        self.m = m
        self.lda=lda
        super().___init___(n, Counter(poissonSample(m, 1, n, lda)),theta,phi,s0)

    sampleType = "Mallows_Poisson"

    def label(self):
        return super().label() + f"_lambda-{lda}"

    def poissonSample(m, lower = 0, upper = float('inf'), lda):
        """Returns m values drawn from a Poisson distribution on lambda for
            which all values are in [lower, upper] through repeated sampling.
            Not guaranteed to terminate.
            Parameters
            ----------
            m: int
                The number of elements in the requested sample
            lower: int
                Lowerbound of the sampled values
            upper: int
                Upperbound of the sampled values.
            lda: float
                The lambda parameter on which the Poisson distribution is
                defined.
        """
        rng = np.random.default_rng()
        # Samples are drawn repeatedly until enough values
        # reside in [lower,upper]
        while True:
            s = rng.poisson(lda, m)
            s = arr[np.where(lower <= s <= upper)]
            if len(s) >= n:
                return s[:n]
            # Double the number of values sampled each time to
            # increase the chance that sufficient values lay in [lower,upper]
            m *= 2


# Main function generates synthetic data-sets using the Mallows-Model
# and saves the output in CSV format. The data-sets are made for the
# the following dispersion parameters: theta = 0, theta = .1
# For each value of theta, two samples are generated, one where all
# lists rank the top 5 of 10 candidates, and another where top-lists have
# their lengths sampled from a Poisson distribution.
# Note: Primarily for testing and debugging.
if __name__ == '__main__':
    m = 20
    n = 10
    thetas = [0, .1]

    for theta in thetas:
        np.savetxt("Theta=%d_top-%d-lists"
            topListSample(n,{5 : m},theta=None,phi=None,s0=None))
