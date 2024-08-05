from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib import pyplot as plt
from rbeval.plot.data import Eval, EvalGroup
from matplotlib import colormaps
from matplotlib.axes import Axes
import numpy as np


def score_cdf(samples: Dict[str, EvalGroup], figure_dir: Path, args: List[str]):
    fig, axs = plt.subplots(
        2, len(samples), figsize=(5 * len(samples), 8), dpi=320, sharey=True
    )
    if len(axs.shape) == 1:
        axs = axs[np.newaxis].T

    for axi, renorm in enumerate([True, False]):
        for ax, (group_name, group) in zip(axs[axi], samples.items()):
            colors = get_color_pallete(group)
            mins, maxs = group.min_fewshots(), group.max_fewshots()

            for m, color in zip(group.model_evals, colors):
                spec = m.eval_spec
                label = None
                if spec.fewshot == maxs[m.model_name]:
                    label = f"{m.model_name} fs{mins[m.model_name]}-{m.model_name}"
                cdf = CdfData.get_correct_prob_cdf(m.evals)
                cdf.plot(ax, color, label)

            ax.set_xlabel("Model output probability")
            ax.set_ylabel("Percent of samples with correct model output prob > p")
            ax.set_title(
                f"{group_name} {'renormed' if renorm else ''} perf curve, mmlu"
            )

            # sort both labels and handles by labels
            ax.legend()

    fig.tight_layout()
    fig.savefig(figure_dir / "score_cdf.png")


def get_color_pallete(group: EvalGroup) -> List[np.ndarray]:
    mins = group.min_fewshots()
    maxs = group.max_fewshots()
    names = set(e.model_name for e in group.model_evals)
    colors = {k: v for k, v in zip(names, ["Greys", "Purples", "Greens", "Reds"])}

    res = []
    for e in group.model_evals:
        vmin, vmax = mins[e.model_name], maxs[e.model_name]
        f = (e.eval_spec.fewshot + 1 - vmin) / (vmax + 1)
        res.append(colormaps[colors[e.model_name]]([f])[0])

    return res


def get_base_logits(probs):
    logits = np.zeros(len(probs))
    logits[0] = 1
    rest = 1 - np.sum(probs)
    assert (rest >= 0) and (rest <= 1)


@dataclass
class CdfData:
    cdf_p: np.ndarray
    scores: np.ndarray

    @classmethod
    def from_weights(cls, weights: np.ndarray, scores: np.ndarray) -> "CdfData":
        sort_perm = scores.argsort()
        sort_weights = weights[sort_perm]
        sort_scores = scores[sort_perm]
        return CdfData(
            cdf_p=1 - np.cumsum(sort_weights),
            scores=sort_scores,
        )

    @classmethod
    def get_correct_prob_cdf(
        cls, evals: List[Eval], prob_renorm=True, sample_weight="stat"
    ):
        """Get's the CDF of the correct probability score

        Args:
            evals (List[Eval]): List of mc evals
            prob_renorm (bool, optional): Whether to renormalize the correct probabilities by the sum of the correct and incorrect. Defaults to False.
            sample_weight (str, optional): Mode to weight samples. Can be either "stat" or "uniform". Defaults to "stat".
        """
        scores = np.concatenate([np.exp(c.cor_logprobs) for c in evals])
        if prob_renorm:
            inc_probs = np.concatenate(
                [np.exp(e.inc_logprobs).sum(axis=1) for e in evals]
            )
            scores /= scores + inc_probs
        num_cats = len(evals)
        weights: np.ndarray
        if sample_weight == "stat":
            arrs = []
            for e in evals:
                n = len(e.cor_logprobs)
                # Each eval gets a total weight of 1/num_cats
                # So each sample should have a weight of 1/num_cats/n
                arrs.append(np.ones(n) / (num_cats * n))
            weights = np.concatenate(arrs)
        else:
            assert sample_weight == "uniform"
            weights = np.ones_like(scores) / len(scores)

        return cls.from_weights(weights, scores)

    def plot(self, ax: Axes, color=None, label: Optional[str] = None):
        ax.plot(self.scores, self.cdf_p, label=label, color=color, alpha=0.8)
        ax.set_xlim(self.scores.min(), self.scores.max())
        ax.set_ylim(0, 1)
