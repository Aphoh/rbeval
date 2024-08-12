from dataclasses import dataclass, field
from typing import List, Optional

from numpy._typing import NDArray
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
        for renorm in [True, False]
        for a in plot_with_data(cfg, get_plot_data(cfg, samples), renorm)
    ]


def get_plot_data(
    cfg: "CdfPlotConfig",
    samples: List[EvalGroup],
) -> PlotData:
    data = PlotData()
    for renorm in [True, False]:
        gfs = data.renorm if renorm else data.norenorm
        for group in samples:
            dfs: List[pd.DataFrame] = []
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
    renorm: bool = True,
) -> List[Figure]:
    figures: List[Figure] = []
    group_dfs = data.renorm if renorm else data.norenorm
    for df in group_dfs:
        group_name: str = str(df["group"].iloc[0])  # type: ignore
        label_selection = alt.selection_point(fields=["label"], bind="legend")  # type: ignore
        fs_selection = alt.selection_point(fields=["fewshot"], bind="legend")  # type: ignore
        chart = (
            alt.Chart(df)  # type: ignore
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=cfg.xlabel),
                y=alt.Y("y:Q", title=cfg.ylabel),
                color=alt.Color(
                    "label:N", legend=alt.Legend(symbolOpacity=1.0, labelLimit=1000)
                ).scale(scheme="set1"),
                opacity=alt.condition(  # type: ignore
                    label_selection & fs_selection,
                    alt.Opacity("fewshot:O"),
                    alt.value(0.0),  # type: ignore
                ),
            )
            .properties(title=cfg.title(group_name, renorm))
            .add_params(fs_selection, label_selection)
            .interactive()
        )
        if cfg.xline is not None:
            line = alt.Chart(pd.DataFrame({"x": [cfg.xline]})).mark_rule().encode(x="x")
            chart = chart + line  # type: ignore
        figures.append(
            Figure(name=f"{group_name} {cfg.name}", chart=chart, group=group_name)
        )

    return figures


class CdfPlotConfig(ABC):
    xlabel: str
    ylabel: str
    name: str = ""
    xline: Optional[float] = None

    @abstractmethod
    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        pass

    def title(self, group_name: str, prob_renorm: bool) -> str:
        title = group_name
        if prob_renorm:
            title += " renormed"
        title += " " + self.name
        return title

    def marks(self) -> alt.Chart:
        raise NotImplementedError()


class CorrectProbCdfPlot(CdfPlotConfig):
    name = "𝚽 Performance Curve"
    xlabel = "𝚽"
    ylabel = "% of correct answers with 𝚽 > x"
    xline = 0.25

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        samples = [np.exp(e.cor_logprobs) for e in evals]
        if prob_renorm:
            samples = [renormed(e)[0] for e in evals]
        return CdfData.from_samples(samples)


class CorrIncorrDiffConfig(CdfPlotConfig):
    name = "𝚫 Performance Curve"
    xline = 0.0
    xlabel = "𝚫"
    ylabel = "% of samples with 𝚫 > x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "CdfData":
        score_arrs: List[NDArray[np.float64]] = []
        for e in evals:
            if prob_renorm:
                cor_probs, inc_probs = renormed(e)
            else:
                cor_probs = np.exp(e.cor_logprobs)
                inc_probs = np.exp(e.inc_logprobs)

            score_arrs.append(cor_probs - inc_probs.max(axis=1))

        return CdfData.from_samples(score_arrs, per_sample_weighting=True)
