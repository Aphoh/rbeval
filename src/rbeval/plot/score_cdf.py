from pathlib import Path
from typing import List

from rbeval.plot.data import Eval, EvalGroup
from matplotlib import colormaps
from matplotlib.axes import Axes
from abc import ABC, abstractmethod
import numpy as np

from rbeval.plot.utils import CdfData, fig_axs_grid, renormed


def score_cdf(samples: List[EvalGroup], figure_dir: Path, args: List[str]):
    plot_with_config(CorrectProbCdfPlot(), samples, figure_dir / "corr_score_cdf.png")
    plot_with_config(
        CorrIncorrDiffConfig(), samples, figure_dir / "cor_incor_gap_cdf_png"
    )


def plot_with_config(
    cfg: "CdfPlotConfig",
    samples: List[EvalGroup],
    figure_out_path: Path,
):
    fig, axs = fig_axs_grid(2, len(samples))
    for i, renorm in enumerate([True, False]):
        for j, group in enumerate(samples):
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
            ax.set_title(cfg.title(group.name, renorm))
            leg = ax.legend()
            for h in leg.legend_handles:
                assert h is not None
                h.set_alpha(1.0)

    fig.tight_layout()
    fig.savefig(str(figure_out_path))


def get_color_pallete(group: EvalGroup) -> List[np.ndarray]:
    mins = group.min_fewshots
    maxs = group.max_fewshots
    names = set(e.model_name for e in group.model_evals)
    cmap_names = ["Greys", "Purples", "Greens", "Reds", "Blues", "Oranges"]
    colors = {k: colormaps[v] for k, v in zip(names, cmap_names)}

    res = []
    for e in group.model_evals:
        vmin, vmax = mins[e.model_name], maxs[e.model_name]
        f = (e.eval_spec.fewshot + 1 - vmin) / (vmax + 2 - vmin)
        res.append(colors[e.model_name]([f])[0])

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
        samples = [np.exp(e.cor_logprobs) for e in evals]
        if prob_renorm:
            samples = [renormed(e)[0] for e in evals]
        return CdfData.from_samples(samples)

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
            if prob_renorm:
                cor_probs, inc_probs = renormed(e)
            else:
                cor_probs = np.exp(e.cor_logprobs)
                inc_probs = np.exp(e.inc_logprobs)

            score_arrs.append(cor_probs - inc_probs.max(axis=1))

        return CdfData.from_samples(score_arrs, per_sample_weighting=True)

    def plot_mark(self, ax: Axes):
        ax.axvline(0.0, linestyle="--", color="red", lw=0.25)
