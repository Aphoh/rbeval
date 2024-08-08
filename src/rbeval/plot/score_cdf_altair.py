from typing import List

from rbeval.plot.data import Eval, EvalGroup, Figure
from abc import ABC, abstractmethod
import numpy as np
import altair as alt
import pandas as pd

from rbeval.plot.utils import CdfData, renormed


def score_cdf(samples: List[EvalGroup], args: List[str]) -> List[Figure]:
    return [
        Figure(
            name="Correct Prob Perf Curve",
            chart=plot_with_config(CorrectProbCdfPlot(), samples),
        ),
        Figure(
            name="Corr-Incorr Gap Perf Curve",
            chart=plot_with_config(CorrIncorrDiffConfig(), samples),
        ),
    ]


def plot_with_config(
    cfg: "CdfPlotConfig",
    samples: List[EvalGroup],
) -> alt.ConcatChart:
    group_dfs = []
    for renorm in [True, False]:
        for group in samples:
            dfs = []
            for m in group.model_evals:
                spec = m.eval_spec
                cdf = cfg.get_cdf(m.evals, renorm)
                df = pd.DataFrame(
                    {
                        "x": cdf.scores,
                        "y": cdf.cdf_p,
                        "label": m.model_name,
                        "renorm": renorm,
                        "fewshot": spec.fewshot,
                    }
                )
                dfs.append(df)
            group_dfs.append(pd.concat(dfs))

    selection = alt.selection_point(fields=["label"], bind="legend")
    charts = []
    for group, df in zip(samples, group_dfs):
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=cfg.xlabel),
                y=alt.Y("y:Q", title=cfg.ylabel),
                color=alt.Color("label:N", legend=alt.Legend(symbolOpacity=1.0)),
                opacity=alt.condition(
                    selection, alt.Opacity("fewshot:O"), alt.value(0.1)
                ),
            )
            .properties(title=cfg.title(group.name, renorm))
            .resolve_legend(color="independent")
        )

        charts.append(chart)

    final_chart = (
        alt.concat(*charts, columns=len(samples)).add_params(selection).interactive()
    )
    return final_chart


class CdfPlotConfig(ABC):
    plot_type: str
    xlabel: str
    ylabel: str

    @abstractmethod
    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        pass

    def title(self, group_name: str, prob_renorm: bool) -> str:
        title = group_name
        if prob_renorm:
            title += " renormed"
        title += " " + self.plot_type
        return title

    def marks(self) -> alt.Chart:
        raise NotImplementedError()


class CorrectProbCdfPlot(CdfPlotConfig):
    def __init__(self):
        self.plot_type = "corr perf plot"
        self.xlabel = "Correct answer probability"
        self.ylabel = "% of correct answers with p > x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        samples = [np.exp(e.cor_logprobs) for e in evals]
        if prob_renorm:
            samples = [renormed(e)[0] for e in evals]
        return CdfData.from_samples(samples)

    def marks(self) -> alt.Chart:
        return (
            alt.Chart(pd.DataFrame({"x": [0.25]}))
            .mark_rule()
            .encode(x="x:Q", color=alt.value("red"))
        )


class CorrIncorrDiffConfig(CdfPlotConfig):
    def __init__(self):
        self.plot_type = "corr-max(incor) perf plot"
        self.xlabel = "corr prob - max(incor prob)"
        self.ylabel = "% of samples with corr - max(incor) > x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        score_arrs = []
        for e in evals:
            if prob_renorm:
                cor_probs, inc_probs = renormed(e)
            else:
                cor_probs = np.exp(e.cor_logprobs)
                inc_probs = np.exp(e.inc_logprobs)

            score_arrs.append(cor_probs - inc_probs.max(axis=1))

        return CdfData.from_samples(score_arrs, per_sample_weighting=True)

    def marks(self) -> alt.Chart:
        return (
            alt.Chart(pd.DataFrame({"x": [0.5]}))
            .mark_rule()
            .encode(x="x:Q", color=alt.value("red"))
        )
