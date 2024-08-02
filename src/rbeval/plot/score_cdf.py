from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot as plt
from rbeval.plot.data import EvalGroup, ModelEval
from matplotlib import colormaps
from matplotlib.axes import Axes
import numpy as np


def score_cdf(samples: Dict[str, EvalGroup], figure_dir: Path, args: List[str]):
    fig, axs = plt.subplots(
        1, len(samples), figsize=(5 * len(samples), 5), dpi=320, sharey=True
    )

    for ax, (group_name, group) in zip(axs, samples.items()):
        group: EvalGroup

        model_names = set(m.model_spec.model_name for m in group.model_evals)
        max_fewshot = {}
        for m in group.model_evals:
            max_fewshot[m.model_name] = max(
                max_fewshot.get(m.model_name, 0), m.model_spec.fewshot
            )

        scales = ["Purples", "Greens", "Oranges", "Reds"]
        model_cmaps = {}
        for i, (scale, n) in enumerate(zip(scales, model_names)):
            mfs = max_fewshot[n]
            if mfs > 0:
                model_cmaps[n] = colormaps[scale](
                    np.linspace(0.4, 1, max_fewshot[n] + 1)
                )
            else:
                model_cmaps[n] = colormaps[scale]([1.0])

        for model_eval in group.model_evals:
            spec = model_eval.model_spec
            color = model_cmaps[spec.model_name][spec.fewshot]
            plot_samples(
                ax, model_eval, model_eval.model_spec.pretty_name(), color=color
            )

        label_ax(ax, title=group_name)

        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

    fig.savefig(figure_dir / "score_cdf.png")


def label_ax(ax, title=True, y=True, x=True):
    if x:
        ax.set_xlabel("Model output probability")
    if y:
        ax.set_ylabel("Percent of samples with correct model output prob > p")
    if title:
        ax.set_title("Performance curve for mmlu")


def get_base_logits(probs):
    logits = np.zeros(len(probs))
    logits[0] = 1
    rest = 1 - np.sum(probs)
    assert (rest >= 0) and (rest <= 1)


def plot_samples(ax: Axes, meval: ModelEval, name: str, norm_by_stat=True, color=None):
    bulk = np.concatenate([np.exp(e.cor_logprobs) for e in meval.evals])
    num_cats = len(meval.evals)
    weights = []
    if norm_by_stat:
        for e in meval.evals:
            n = len(e.cor_logprobs)
            # Each eval gets a total weight of 1/num_cats
            # So each sample should have a weight of 1/num_cats/n
            weights.append(np.ones(n) / (num_cats * n))
        weights = np.concatenate(weights)
    else:
        weights = np.ones_like(bulk) / len(bulk)

    sort_perm = bulk.argsort()
    bulk = bulk[sort_perm]
    weights = weights[sort_perm]
    cdf_p = 1 - np.cumsum(weights)

    ax.plot(bulk, cdf_p, label=name, color=color, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
