from dataclasses import dataclass
from typing import List, Optional
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from rbeval.plot.data import Eval


def fig_axs_grid(
    n_rows: int, n_cols: int, sharex=False, sharey=True
) -> tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=320,
        sharey=sharey,
        sharex=sharex,
    )
    if isinstance(axs, Axes):
        axs = np.array([[axs]])
    if len(axs.shape) == 1:
        axs = axs[np.newaxis].T
    return fig, axs


def renormed(eval: Eval) -> tuple[np.ndarray, np.ndarray]:
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
    def from_samples(cls, samples: List[np.ndarray], per_sample_weighting=True):
        num_cats = len(samples)
        scores = np.concatenate(samples)
        if per_sample_weighting:
            weight_arrs = []
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
        cls, weights: np.ndarray, scores: np.ndarray, max_p=600
    ) -> "CdfData":
        sort_perm = scores.argsort()
        base_weights = weights[sort_perm]
        base_scores = scores[sort_perm]
        base_cdf_p = 1 - np.cumsum(base_weights)
        minscore, maxscore = base_scores[0], base_scores[-1]
        if len(base_scores) > max_p:
            scores = np.linspace(minscore, maxscore, max_p)
            cdf_p = np.interp(scores, base_scores, base_cdf_p)
        else:
            scores, cdf_p = base_scores, base_cdf_p
        return CdfData(
            cdf_p=cdf_p,
            scores=scores,
        )

    def plot(self, ax: Axes, color=None, label: Optional[str] = None):
        ax.plot(self.scores, self.cdf_p, label=label, color=color, alpha=0.40)
        ax.set_xlim(self.scores.min(), self.scores.max())
        ax.set_ylim(0, 1)
