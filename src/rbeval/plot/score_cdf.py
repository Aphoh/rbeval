from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib import pyplot as plt
from rbeval.plot.data import Eval, EvalGroup
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from abc import ABC, abstractmethod
import numpy as np


def fig_axs_grid(n_rows: int, n_cols: int) -> tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=320,
        sharey=True,
        sharex=False,
    )
    if len(axs.shape) == 1:
        axs = axs[np.newaxis].T
    return fig, axs


def plot_with_config(
    cfg: "CdfPlotConfig",
    samples: Dict[str, EvalGroup],
    figure_out_path: Path,
):
    fig, axs = fig_axs_grid(2, len(samples))
    for i, renorm in enumerate([True, False]):
        for j, (group_name, group) in enumerate(samples.items()):
            ax = axs[i, j]
            assert isinstance(ax, Axes)
            colors = get_color_pallete(group)
            for m, color in zip(group.model_evals, colors):
                spec = m.eval_spec
                label = (
                    group.model_label(m.model_name)
                    if spec.fewshot == group.max_fewshots[m.model_name]
                    else None
                )
                cdf = cfg.get_cdf(m.evals, renorm)
                cdf.plot(ax, color, label)
                cfg.plot_mark(ax)

            if i == axs.shape[0] - 1:
                ax.set_xlabel(cfg.xlabel)
            if j == 0:
                ax.set_ylabel(cfg.ylabel)
            ax.set_title(cfg.title(group_name, renorm))
            ax.legend()

    fig.tight_layout()
    fig.savefig(str(figure_out_path))


def score_cdf(samples: Dict[str, EvalGroup], figure_dir: Path, args: List[str]):
    plot_with_config(CorrectProbCdfPlot(), samples, figure_dir / "corr_score_cdf.png")
    plot_with_config(
        CorrIncorrDiffConfig(), samples, figure_dir / "cor_incor_gap_cdf_png"
    )


def get_color_pallete(group: EvalGroup) -> List[np.ndarray]:
    mins = group.min_fewshots
    maxs = group.max_fewshots
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


class CdfPlotConfig(ABC):
    plot_type: str
    xlabel: str
    ylabel: str

    @abstractmethod
    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        pass

    def plot_mark(self, ax: Axes):
        pass

    def title(self, group_name: str, prob_renorm: bool) -> str:
        title = group_name
        if prob_renorm:
            title += " renormed"
        title += " " + self.plot_type
        return title


class CorrectProbCdfPlot(CdfPlotConfig):
    def __init__(self):
        self.plot_type = "corr perf plot"
        self.xlabel = "Correct answer probability"
        self.ylabel = "% of correct answers with p > x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
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
        arrs = []
        for e in evals:
            n = len(e.cor_logprobs)
            # Each eval gets a total weight of 1/num_cats
            # So each sample should have a weight of 1/num_cats/n
            arrs.append(np.ones(n) / (num_cats * n))
        weights = np.concatenate(arrs)
        # weights = np.ones_like(scores) / len(scores) TODO: config sample weight probabilities

        return CdfData.from_weights(weights, scores)

    def plot_mark(self, ax: Axes):
        ax.axvline(0.25, linestyle="--", color="red", lw=0.25)
        ax.axhline(0.75, linestyle="--", color="red", lw=0.25)


class CorrIncorrDiffConfig(CdfPlotConfig):
    def __init__(self):
        self.plot_type = "corr-max(incor) perf plot"
        self.xlabel = "corr prob - max(incor prob)"
        self.ylabel = "% of samples with corr - max(incor) > x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        """Get's the CDF of the difference between correct probability and highest incorrect probability"""
        score_arrs = []
        for e in evals:
            cor_probs = np.exp(e.cor_logprobs)
            inc_probs = np.exp(e.inc_logprobs)
            if prob_renorm:
                tot_probs = cor_probs + inc_probs.sum(axis=1)
                cor_probs /= tot_probs
                inc_probs /= tot_probs[np.newaxis].T

            highest_inc_probs = np.max(inc_probs, axis=1)
            diff = cor_probs - highest_inc_probs
            score_arrs.append(diff)

        scores = np.concatenate(score_arrs)

        num_cats = len(evals)
        weight_arrs = []
        for e in evals:
            n = len(e.cor_logprobs)
            # Each eval gets a total weight of 1/num_cats
            # So each sample should have a weight of 1/num_cats/n
            weight_arrs.append(np.ones(n) / (num_cats * n))
        weights = np.concatenate(weight_arrs)
        # weights = np.ones_like(scores) / len(scores) TODO: config sample weight probabilities

        return CdfData.from_weights(weights, scores)

    def plot_mark(self, ax: Axes):
        ax.axvline(0.0, linestyle="--", color="red", lw=0.25)


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

    def plot(self, ax: Axes, color=None, label: Optional[str] = None, max_pts=800000):
        minscore, maxscore = self.scores.min(), self.scores.max()
        if len(self.cdf_p) > max_pts:
            x = np.linspace(minscore, maxscore, max_pts)
            y = np.interp(x, self.scores, self.cdf_p)
        else:
            x, y = self.scores, self.cdf_p

        ax.plot(x, y, label=label, color=color, alpha=0.40)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1)
