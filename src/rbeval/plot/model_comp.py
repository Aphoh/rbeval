import argparse
import altair as alt
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
import itertools
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np

from rbeval.eval_spec import EvalSpec
from rbeval.plot.data import EvalGroup, ModelEval
from rbeval.plot.utils import CdfData, renormed


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
    for fewshot in sorted(fewshots):
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
    charts = []
    for title, score_list in grouped.items():
        diff_cdf_data = []
        corr_cdf_data = []
        for score in score_list:
            diff_cdf = CdfData.from_samples(score.cor_minus_inc_samples)
            diff_cdf_data.append(
                pd.DataFrame(
                    {
                        "p": diff_cdf.scores,
                        "1-CDF(p)": diff_cdf.cdf_p,
                        "fewshot": score.spec.fewshot,
                        "model": score.spec.model_name,
                    }
                )
            )
            corr_cdf = CdfData.from_samples(score.cor_samples)
            corr_cdf_data.append(
                pd.DataFrame(
                    {
                        "p": corr_cdf.scores,
                        "1-CDF(p)": corr_cdf.cdf_p,
                        "fewshot": score.spec.fewshot,
                        "model": score.spec.model_name,
                    }
                )
            )

        diff_cdf_df = pd.concat(diff_cdf_data)
        corr_cdf_df = pd.concat(corr_cdf_data)
        diff_cdf_chart = (
            alt.Chart(diff_cdf_df)
            .mark_line()
            .encode(
                x=alt.X("p:Q"),
                y=alt.Y("1-CDF(p):Q"),
                opacity=alt.Opacity("fewshot:Q"),
                color=alt.Color("model:N"),
            )
            .properties(title=f"{title}, tuned - base", width=300, height=200)
        )
        corr_cdf_chart = (
            alt.Chart(corr_cdf_df)
            .mark_line()
            .encode(
                x=alt.X("p:Q"),
                y=alt.Y("1-CDF(p):Q"),
                opacity=alt.Opacity("fewshot:Q"),
                color=alt.Color("model:N"),
            )
            .properties(title=f"{title}, tuned cor - max(inc)", width=300, height=200)
        )

        chart = alt.vconcat(diff_cdf_chart, corr_cdf_chart)
        charts.append(chart)

    final_chart = alt.hconcat(*charts)
    final_chart.save(str(figure_path))


def plot_by_group(grouped: Dict[str, List[Scores]], figure_path: Path):
    charts = []

    for title, scores in grouped.items():
        data = []
        for s in scores:
            cnt = sum(len(a) for a in s.cor_minus_inc_samples)
            fs = s.spec.fewshot
            data.append({"Fewshot": fs, "Count": cnt, "Model": s.spec.model_name})

        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Fewshot:O", title="Fewshot"),
                y=alt.Y("Count:Q", title="Number of Samples"),
                color="Model:N",
                tooltip=["Fewshot", "Count"],
            )
            .properties(title=f"{title} counts")
        )
        charts.append(chart)

    final_chart = (
        alt.hconcat(*charts).resolve_axis(y="shared").resolve_scale(y="shared")
    )
    final_chart.save(str(figure_path))


def plot_by_fewshot(grouped: Dict[str, List[Scores]], figure_path: Path):
    fewshot_grouped: Dict[int, List[tuple[str, Scores]]] = defaultdict(list)
    for title, score_list in grouped.items():
        for s in score_list:
            fewshot_grouped[s.spec.fewshot].append((title, s))

    charts = []
    for fs, title_score_pairs in sorted(fewshot_grouped.items()):
        data = []
        for title, scores in title_score_pairs:
            cnt = sum(len(a) for a in scores.cor_minus_inc_samples)
            data.append(
                {
                    "Title": title,
                    "Count": cnt,
                    "Fewshot": fs,
                    "Model": scores.spec.model_name,
                }
            )

        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Title:O", title="Categories"),
                y=alt.Y("Count:Q", title="Number of Samples"),
                tooltip=["Title", "Count", "Fewshot"],
                color="Model:N",
            )
            .properties(title=f"Fewshot {fs}")
        )
        charts.append(chart)

    final_chart = (
        alt.hconcat(*charts).resolve_scale(y="shared").resolve_axis(y="shared")
    )
    final_chart.save(str(figure_path), scale_factor=2.0)


def by_fewshot(model_evals: List[ModelEval]):
    fewshot_counts: Dict[int, List[ModelEval]] = defaultdict(list)
    for model_eval in model_evals:
        fewshot_counts[model_eval.eval_spec.fewshot].append(model_eval)
    return {k: v for k, v in fewshot_counts.items()}
