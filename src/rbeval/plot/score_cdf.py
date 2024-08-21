from typing import List, Literal, Optional

from numpy._typing import NDArray
from rbeval.plot.data import Eval, EvalGroup, Figure
from abc import ABC, abstractmethod
import numpy as np
import altair as alt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score  # type: ignore

from rbeval.plot.utils import PlotData, renormed


def plot_cfgs():
    return [
        CorrectProbCdfPlot(),
        CorrIncorrDiffConfig(),
        ROCCurve(),
        MaxIncorProbCdfPlot(),
        AccVsLoss(),
        AccVsAUC(),
    ]


def score_cdf(samples: List[EvalGroup], args: List[str]) -> List[Figure]:
    return [
        a
        for cfg in plot_cfgs()
        for a in plot_with_data(cfg, get_plot_data(cfg, samples))
    ]


def get_plot_data(
    cfg: "CdfPlotConfig",
    samples: List[EvalGroup],
) -> pd.DataFrame:
    records = []
    for renorm in [True, False]:
        for group in samples:
            for m in group.model_evals:
                spec = m.eval_spec
                cdf = cfg.get_cdf(m.evals, renorm)
                records.append(
                    {
                        "x": cdf.x,
                        "y": cdf.y,
                        "label": m.model_name,
                        "group": group.name,
                        "renorm": renorm,
                        "fewshot": spec.fewshot,
                    }
                )
    data = pd.DataFrame.from_records(records)
    return data


def plot_with_data(
    cfg: "CdfPlotConfig",
    data: pd.DataFrame,
) -> List[Figure]:
    figures: List[Figure] = []
    for (group_name, renorm), df in data.groupby(["group", "renorm"]):
        assert isinstance(group_name, str)
        assert isinstance(renorm, (bool, np.bool_))
        label_selection = alt.selection_point(fields=["label"], bind="legend")  # type: ignore
        fs_selection = alt.selection_point(fields=["fewshot"], bind="legend")  # type: ignore
        chart = alt.Chart(df.explode(["x", "y"]))  # type: ignore
        chart = chart.mark_line() if cfg.type == "line" else chart.mark_point()
        chart = (
            chart.encode(
                x=alt.X("x:Q", title=cfg.xlabel, scale=alt.Scale(zero=False)),
                y=alt.Y("y:Q", title=cfg.ylabel, scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "label:N", legend=alt.Legend(symbolOpacity=1.0, labelLimit=1000)
                ).scale(scheme="dark2"),
                shape="label:N" if cfg.type == "scatter" else alt.Undefined,
                opacity=alt.condition(  # type: ignore
                    label_selection & fs_selection,
                    alt.Opacity("fewshot:O"),
                    alt.value(0.0),  # type: ignore
                ),
            )
            .properties(title=cfg.title(group_name, renorm))  # type: ignore
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
    type: Literal["line", "scatter"] = "line"

    @abstractmethod
    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
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
    name = "CDF(𝚽)"
    xlabel = "𝚽"
    ylabel = "% of correct answers with 𝚽 < x"
    xline = 0.25

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
        samples = [np.exp(e.cor_logprobs) for e in evals]
        if prob_renorm:
            samples = [renormed(e)[0] for e in evals]
        return PlotData.perf_curve_from_samples(samples)


class MaxIncorProbCdfPlot(CdfPlotConfig):
    name = "CDF(Max(Incorrect))"
    xlabel = "max(incorrect)"
    ylabel = "% of correct answers with max(incorrect) < x"
    xline = 0.25

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
        if prob_renorm:
            samples = [renormed(e)[1].max(axis=1) for e in evals]
        else:
            samples = [np.exp(np.max(e.inc_logprobs, axis=1)) for e in evals]
        return PlotData.perf_curve_from_samples(samples)


class AccVsLoss(CdfPlotConfig):
    name = "Cross Entropy Loss vs Accuracy"
    xlabel = "Accuracy"
    ylabel = "CE Loss"
    xline = None
    type = "scatter"

    def get_cdf(self, evals: List[Eval], _prob_renorm: bool) -> "PlotData":
        cor, incor = zip(*[renormed(e) for e in evals])
        cor = np.concatenate(cor)
        incor = np.concatenate(incor).max(axis=1)
        pct_corr = np.mean(cor > incor)

        celoss = np.mean(-np.log(cor))
        return PlotData(np.array([celoss]), np.array([pct_corr]))


class AccVsAUC(CdfPlotConfig):
    name = "Simulated AUROC vs Accuracy"
    xlabel = "Accuracy"
    ylabel = "Simulated AUROC"
    xline = None
    type = "scatter"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
        cor, incor = zip(*[renormed(e) for e in evals])
        cor = np.concatenate(cor)
        incor = np.concatenate(incor).max(axis=1)
        pct_corr = np.mean(cor > incor)

        scores, labels, weights = roc_data(evals, prob_renorm)
        auc = roc_auc_score(labels, scores, sample_weight=weights)
        return PlotData(np.array([auc]), np.array([pct_corr]))


class CorrIncorrDiffConfig(CdfPlotConfig):
    name = "CDF(𝚫)"
    xline = 0.0
    xlabel = "𝚫"
    ylabel = "% of samples with 𝚫 < x"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
        score_arrs: List[NDArray[np.float64]] = []
        for e in evals:
            if prob_renorm:
                cor_probs, inc_probs = renormed(e)
            else:
                cor_probs = np.exp(e.cor_logprobs)
                inc_probs = np.exp(e.inc_logprobs)

            score_arrs.append(cor_probs - inc_probs.max(axis=1))

        return PlotData.perf_curve_from_samples(score_arrs, per_sample_weighting=True)


class ROCCurve(CdfPlotConfig):
    name = "Simulated ROC Curve"
    xline = None
    xlabel = "FPR"
    ylabel = "TPR"

    def get_cdf(self, evals: List[Eval], prob_renorm: bool) -> "PlotData":
        scores, labels, weights = roc_data(evals, prob_renorm)
        assert len(scores) == len(labels) == len(weights)
        tpr, fpr, _ = roc_curve(labels, scores, sample_weight=weights)

        x_interp = np.linspace(0, 1, 600)
        y_interp = np.interp(x_interp, fpr, tpr)

        return PlotData(x_interp, y_interp)


def roc_data(evals: List[Eval], prob_renorm):
    weight_arrs = []
    total = sum(len(e.cor_logprobs) for e in evals)
    for samples in evals:
        this = np.ones(2 * len(samples.cor_logprobs)) / (2 * total)
        weight_arrs.append(this)

    score_arrs = []
    label_arrs = []
    for e in evals:
        if prob_renorm:
            cor_probs, inc_probs = renormed(e)
        else:
            cor_probs = np.exp(e.cor_logprobs)
            inc_probs = np.exp(e.inc_logprobs)
        score_arrs.append(cor_probs)
        label_arrs.append(np.ones(len(cor_probs)))
        score_arrs.append(inc_probs.max(axis=1))
        label_arrs.append(np.zeros(inc_probs.shape[0]))

    scores = np.concatenate(score_arrs)
    labels = np.concatenate(label_arrs)
    weights = np.concatenate(weight_arrs)
    return scores, labels, weights
