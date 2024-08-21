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
class PlotData:
    y: np.ndarray
    x: np.ndarray

    @classmethod
    def perf_curve_from_samples(
        cls,
        samples: List[NDArray[np.float64]],
        per_sample_weighting: bool = True,
        one_minus: bool = False,
    ) -> "PlotData":
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
        return cls.perf_curve_from_weights(weights, scores, one_minus=one_minus)

    @classmethod
    def perf_curve_from_weights(
        cls,
        weights: NDArray[np.float64],
        base_scores: NDArray[np.float64],
        max_p: int = 600,
        one_minus: bool = True,
    ) -> "PlotData":
        sort_perm = base_scores.argsort()
        base_weights = weights[sort_perm]
        base_scores = base_scores[sort_perm]
        base_cdf_p = np.cumsum(base_weights)
        if one_minus:
            base_cdf_p = 1 - base_cdf_p
        minscore, maxscore = base_scores[0], base_scores[-1]
        if len(base_scores) > max_p:
            scores = np.linspace(minscore, maxscore, max_p)  # type: ignore
            cdf_p = np.interp(scores, base_scores, base_cdf_p)  # type: ignore
        else:
            scores, cdf_p = base_scores, base_cdf_p
        return PlotData(
            y=cdf_p,
            x=scores,  # type: ignore
        )
