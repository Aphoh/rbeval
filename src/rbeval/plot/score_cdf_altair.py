from dataclasses import dataclass, field
from typing import List

from rbeval.plot.data import Eval, EvalGroup, Figure
from abc import ABC, abstractmethod
import numpy as np
import altair as alt
import pandas as pd

from rbeval.plot.utils import CdfData, renormed


@dataclass
class PlotData:
    renorm: List[pd.DataFrame] = field(default_factory=list)
    norenorm: List[pd.DataFrame] = field(default_factory=list)


def plot_cfgs():
    return [CorrectProbCdfPlot(), CorrIncorrDiffConfig()]


def score_cdf(samples: List[EvalGroup], args: List[str]) -> List[Figure]:
    return [
        a
        for cfg in plot_cfgs()
        for a in plot_with_data(cfg, get_plot_data(cfg, samples))
    ]


def get_plot_data(
    cfg: "CdfPlotConfig",
    samples: List[EvalGroup],
) -> PlotData:
    data = PlotData()
    for renorm in [True, False]:
        gfs = data.renorm if renorm else data.norenorm
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
                        "group": group.name,
                        "renorm": renorm,
                        "fewshot": spec.fewshot,
                    }
                )
                dfs.append(df)
            gfs.append(pd.concat(dfs))
    return data


def plot_with_data(
    cfg: "CdfPlotConfig",
    data: PlotData,
) -> List[Figure]:
    figures = []
    for renorm, group_dfs in zip([True, False], [data.renorm, data.norenorm]):
        for df in group_dfs:
            group_name = df["group"].iloc[0]
            selection = alt.selection_point(fields=["label"], bind="legend")
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
                .properties(title=cfg.title(group_name, renorm), width=800, height=400)
                .resolve_legend(color="independent")
                .resolve_axis(y="independent", x="independent")
                .add_params(selection)
                .interactive()
            )
            figures.append(Figure(name=f"{group_name} {cfg.name}", chart=chart))

    return figures


class CdfPlotConfig(ABC):
    plot_type: str
    xlabel: str
    ylabel: str
    name: str = ""

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
    name = "Correct Prob Perf Curve"

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
    name = "Corr-Incorr Gap Perf Curve"

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
