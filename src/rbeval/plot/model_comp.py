import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import itertools
from pathlib import Path
from matplotlib import colormaps
from typing import Dict, List, Optional
import warnings

import numpy as np

from rbeval.eval_spec import EvalSpec
from rbeval.plot.data import EvalGroup, ModelEval
from rbeval.plot.utils import CdfData, fig_axs_grid, renormed


@dataclass
class Scores:
    spec: EvalSpec
    cor_minus_inc_samples: List[np.ndarray] = field(default_factory=list)
    cor_samples: List[np.ndarray] = field(default_factory=list)


def model_comparer(samples: List[EvalGroup], figure_dir: Path, rem_args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--compare", type=str)
    args = parser.parse_args(rem_args)
    base_name_filt: Optional[str] = args.base
    comp_name_filt: Optional[str] = args.compare

    if base_name_filt is None or comp_name_filt is None:
        warnings.warn(
            "Skipping model comparison plot, need to specify base and compare"
        )
        return

    bases: List[ModelEval] = list(
        itertools.chain.from_iterable(
            g.collect_with_name(base_name_filt) for g in samples
        )
    )
    comps: List[ModelEval] = list(
        itertools.chain.from_iterable(
            g.collect_with_name(comp_name_filt) for g in samples
        )
    )
    bases_mnames = set(e.model_name for e in bases)
    comps_mnames = set(e.model_name for e in comps)

    assert len(bases_mnames) == 1, f"Got multiple base models: {bases_mnames}"
    assert len(comps_mnames) == 1, f"Got multiple comp models: {comps_mnames}"

    base_name = bases_mnames.pop()
    comp_name = comps_mnames.pop()

    bases_fs = by_fewshot(bases)
    for k, v in bases_fs.items():
        if len(v) > 1:
            print(f"Duplicate base fewshot {k} for {[e.model_name for e in v]}")
    compares_fs = by_fewshot(comps)
    for k, v in compares_fs.items():
        if len(v) > 1:
            print(f"Duplicate compare fewshot {k} for {[e.model_name for e in v]}")

    diff_fs = set(bases_fs.keys()).symmetric_difference(compares_fs.keys())
    if len(diff_fs) != 0:
        print(f"Got different fewshots for base and compare {diff_fs}")

    fewshots = set(bases_fs.keys()).intersection(compares_fs.keys())
    grouped: Dict[str, List[Scores]] = defaultdict(list)
    for i, fewshot in enumerate(sorted(fewshots)):
        base_eval = bases_fs[fewshot][0]
        compare_eval = compares_fs[fewshot][0]
        base_by_category = {e.task_name(): e for e in base_eval.evals}
        comp_by_category = {e.task_name(): e for e in compare_eval.evals}

        assert set(base_by_category.keys()) == set(comp_by_category.keys())
        scores_by_mask: Dict[str, Scores] = defaultdict(
            lambda: Scores(compare_eval.eval_spec)
        )
        for k in base_by_category.keys():
            base = base_by_category[k]
            comp = comp_by_category[k]
            base_inc_mask = base.inc_logprobs.max(axis=1) > base.cor_logprobs
            base_cor_mask = base.cor_logprobs > base.inc_logprobs.max(axis=1)
            comp_inc_mask = comp.inc_logprobs.max(axis=1) > comp.cor_logprobs
            comp_cor_mask = comp.cor_logprobs > comp.inc_logprobs.max(axis=1)

            for title, mask in [
                ("1 Inc -> Inc", base_inc_mask & comp_inc_mask),
                ("2 Inc -> Cor", base_inc_mask & comp_cor_mask),
                ("3 Cor -> Inc", base_cor_mask & comp_inc_mask),
                ("4 Cor -> Cor", base_cor_mask & comp_cor_mask),
            ]:
                base_ev = base.filter_mask(mask)
                comp_ev = comp.filter_mask(mask)
                base_cor, base_inc = renormed(base_ev)
                comp_cor, comp_inc = renormed(comp_ev)
                cor_diff = comp_cor - base_cor
                inc_diff = comp_inc.max(axis=1) - base_inc.max(axis=1)
                scores_by_mask[title].cor_minus_inc_samples.append(cor_diff - inc_diff)
                scores_by_mask[title].cor_samples.append(
                    comp_cor - comp_inc.max(axis=1)
                )

        for title, scores in scores_by_mask.items():
            grouped[title].append(scores)

    plot_diff_cdf(grouped, figure_dir / f"{base_name}_to_{comp_name}_diff_cdf.png")
    plot_by_group(grouped, figure_dir / f"{base_name}_to_{comp_name}_cnt_by_group.png")
    plot_by_fewshot(
        grouped, figure_dir / f"{base_name}_to_{comp_name}_cnt_by_fewshot.png"
    )


def plot_diff_cdf(grouped: Dict[str, List[Scores]], figure_path: Path):
    fig, axs = fig_axs_grid(2, len(grouped))
    for i, title in enumerate(sorted(grouped.keys())):
        score_list = grouped[title]
        fewshots = set(s.spec.fewshot for s in score_list)
        min_fs, max_fs = min(fewshots), max(fewshots)
        for score in score_list:
            label = None
            if score.spec.fewshot == max_fs:
                label = f"{score.spec.model_name} fs{min_fs}-{max_fs}"
            color = colormaps["Oranges"](
                (score.spec.fewshot + 1 - min_fs) / (max_fs + 1 - min_fs)
            )

            diff_cdf = CdfData.from_samples(score.cor_minus_inc_samples)
            diff_cdf.plot(axs[0, i], label=label, color=color)

            corr_cdf = CdfData.from_samples(score.cor_samples)
            corr_cdf.plot(axs[1, i], label=label, color=color)

        axs[0, i].set_title(f"{title}, tuned - base")
        axs[0, i].set_xlabel(
            "p=tuned_cor - max(tuned_inc) - (base_cor - max(base_inc))"
        )
        axs[0, i].legend()

        axs[1, i].set_title(f"{title} tuned cor - max(inc)")
        axs[1, i].set_xlabel("p=tuned_cor - max(tuned_inc)")
        axs[1, i].legend()

    axs[0, 0].set_ylabel("1-CDF(p)")
    axs[1, 0].set_ylabel("1-CDF(p)")

    fig.tight_layout()
    fig.savefig(str(figure_path))


def plot_by_group(grouped: Dict[str, List[Scores]], figure_path: Path):
    fig, ax = fig_axs_grid(1, 4, sharey=False)
    for i, (title, scores) in enumerate(grouped.items()):
        for s in scores:
            cnt = sum(len(a) for a in s.cor_minus_inc_samples)
            fs = s.spec.fewshot
            ax[i, 0].bar(fs, cnt, label=f"fs{fs}")
            ax[i, 0].set_title(f"{title} counts")
            ax[i, 0].set_xlabel("Fewshot")
    ax[0, 0].set_ylabel("Number of samples")

    fig.tight_layout()
    fig.savefig(str(figure_path))


def plot_by_fewshot(grouped: Dict[str, List[Scores]], figure_path: Path):
    fewshot_grouped: Dict[int, List[tuple[str, Scores]]] = defaultdict(list)
    for title, score_list in grouped.items():
        for s in score_list:
            fewshot_grouped[s.spec.fewshot].append((title, s))

    fewshots = sorted(fewshot_grouped.keys())
    fig, ax = fig_axs_grid(1, len(fewshots))
    for i, (fs, title_score_pairs) in enumerate(fewshot_grouped.items()):
        for title, scores in title_score_pairs:
            cnt = sum(len(a) for a in scores.cor_minus_inc_samples)
            ax[i, 0].bar(title, cnt, label=f"fs{fs}")
            ax[i, 0].set_xlabel("Categories")
    ax[0, 0].set_ylabel("Number of samples")

    fig.tight_layout()
    fig.savefig(str(figure_path))


def by_fewshot(model_evals: List[ModelEval]):
    fewshot_counts: Dict[int, List[ModelEval]] = defaultdict(list)
    for model_eval in model_evals:
        fewshot_counts[model_eval.eval_spec.fewshot].append(model_eval)
    return {k: v for k, v in fewshot_counts.items()}
