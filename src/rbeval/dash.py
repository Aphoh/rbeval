from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import streamlit as st
import argparse
from dacite import from_dict

from rbeval.plot.data import EvalGroup, get_samples
from rbeval.plot.score_cdf import (
    plot_with_data,
    get_plot_data,
    plot_cfgs,
)
from rbeval.plot import model_comp
from huggingface_hub import snapshot_download  # type: ignore


@st.cache_resource
def cached_samples(dir: Path, name_filter: Optional[str]) -> List[EvalGroup]:
    if not dir.exists():
        dir = Path(snapshot_download("mli-will/rbeval"))
    samples = get_samples(dir, name_filter)
    return samples


@st.cache_data
def cached_score_cdf(dir: Path, name_filter: Optional[str]):
    samples = cached_samples(dir, name_filter)
    cfgs = plot_cfgs()
    data = [get_plot_data(cfg, samples) for cfg in cfgs]
    return data, cfgs


@st.cache_data
def cache_compare(
    dir: Path, name_filter: Optional[str], base_name: str, compare_name: str
):
    samples = cached_samples(dir, name_filter)
    grouped, base_name, comp_name = model_comp.get_scores(
        samples, base_name + "$", compare_name + "$"
    )
    grouped_dict = {k: [asdict(vi) for vi in v] for k, v in grouped.items()}
    return grouped_dict, base_name, comp_name


def main():
    parser = argparse.ArgumentParser(description="rbeval dashboard")
    parser.add_argument("--evals", type=str, default="./lmo-fake", required=False)
    args, _rest = parser.parse_known_args()
    eval_dir = Path(args.evals)
    # Show all the models
    st.set_page_config(layout="wide")
    score_cdf_data, cfgs = cached_score_cdf(eval_dir, None)
    for data, cfg in zip(score_cdf_data, cfgs):
        figs = plot_with_data(cfg, data)
        with st.expander(cfg.name):
            for fig in figs:
                st.altair_chart(fig.chart)  # type: ignore

    model_names = set(
        [
            m.model_name
            for group in cached_samples(eval_dir, None)
            for m in group.model_evals
        ]
    )
    base_model = st.selectbox("Base model", model_names)
    compare_model = st.selectbox("Compare model", model_names)
    st.text(f"Comparing {base_model} with {compare_model}")
    if base_model and compare_model:
        if base_model == compare_model:
            st.text("Base and compare models are the same")
            return
        grouped, base_name, comp_name = cache_compare(
            eval_dir, None, base_model, compare_model
        )
        grouped = {
            k: [from_dict(model_comp.Scores, vi) for vi in v]
            for k, v in grouped.items()
        }
        for fig in model_comp.get_figures(grouped, base_name, comp_name):
            st.text(fig.name)
            st.altair_chart(fig.chart)  # type: ignore


if __name__ == "__main__":
    main()
