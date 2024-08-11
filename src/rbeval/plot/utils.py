from dataclasses import dataclass
from typing import List
import numpy as np

from numpy.typing import NDArray
from rbeval.plot.data import Eval


def renormed(eval: Eval) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns tuple (cor, inc) of normalized probabilities"""
    cor = np.exp(eval.cor_logprobs)
    inc = np.exp(eval.inc_logprobs)
    tot = cor + inc.sum(axis=1)
    cor /= tot
    inc /= tot[np.newaxis].T
    return cor, inc


@dataclass
class CdfData:
    cdf_p: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_samples(
        cls, samples: List[NDArray[np.float64]], per_sample_weighting: bool = True
    ) -> "CdfData":
        num_cats = len(samples)
        scores = np.concatenate(samples)
        if per_sample_weighting:
            weight_arrs: List[NDArray[np.float64]] = []
            for s in samples:
                n = len(s)
                # Each eval gets a total weight of 1/num_cats
                # So each sample should have a weight of 1/num_cats/n
                weight_arrs.append(np.ones(n) / (num_cats * n))
            weights = np.concatenate(weight_arrs)
        else:
            weights = np.ones_like(scores) / len(scores)
        return cls.from_weights(weights, scores)

    @classmethod
    def from_weights(
        cls,
        weights: NDArray[np.float64],
        base_scores: NDArray[np.float64],
        max_p: int = 600,
    ) -> "CdfData":
        sort_perm = base_scores.argsort()
        base_weights = weights[sort_perm]
        base_scores = base_scores[sort_perm]
        base_cdf_p = 1 - np.cumsum(base_weights)
        minscore, maxscore = base_scores[0], base_scores[-1]
        if len(base_scores) > max_p:
            scores = np.linspace(minscore, maxscore, max_p)  # type: ignore
            cdf_p = np.interp(scores, base_scores, base_cdf_p)  # type: ignore
        else:
            scores, cdf_p = base_scores, base_cdf_p
        return CdfData(
            cdf_p=cdf_p,
            scores=scores,  # type: ignore
        )
