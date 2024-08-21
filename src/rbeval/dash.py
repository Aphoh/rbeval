from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import pandas as pd
import streamlit as st
import argparse
from dacite import from_dict

from rbeval.plot.dash_utils import markdown_insert_images
from rbeval.plot.data import EvalGroup, get_samples
from rbeval.plot.score_cdf import (
    CdfPlotConfig,
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
def cached_score_cdf(
    dir: Path, name_filter: Optional[str]
) -> tuple[List[pd.DataFrame], List[CdfPlotConfig]]:
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

    with st.expander("README", expanded=False):
        with open("README.md", "r") as f:
            markdown = f.read().split("---", 2)[-1]
            st.markdown(markdown_insert_images(markdown), unsafe_allow_html=True)

    score_cdf_data, cfgs = cached_score_cdf(eval_dir, None)
    assert len(score_cdf_data) > 0, "No score cdfs found"
    group_names: List[str] = sorted(
        score_cdf_data[0]["group"].unique().tolist(), reverse=True
    )

    st.markdown("""
    Below is a toggle which renormalizes the multiple choice answer probabilities to sum to 1.
    For more performant models (anything after Llama 1) or in higher fewshot scenarios, this doesn't impact the results very much.
    """)

    renormed = st.toggle("Renormalize Probabilities", True)
    # fs_names = [str(i) + "-shot" for i in range(0, 5 + 1)]
    # fs_filt_sel = st.multiselect("Fewshot Filter", fs_names, default=fs_names)
    # fs_filt = [int(i.split("-")[0]) for i in fs_filt_sel]
    fs_filt = [i for i in range(0, 5 + 1) if st.checkbox(f"{i}-shot", True)]

    st.subheader("Model Performance Curves")
    for group in group_names:
        with st.expander(group):
            for cfg, df in zip(cfgs, score_cdf_data):
                group_data = df[
                    (df["group"] == group)
                    & (df["renorm"] == renormed)
                    & (df["fewshot"].isin(fs_filt))
                ]
                for fig in plot_with_data(cfg, group_data):
                    st.altair_chart(fig.chart, use_container_width=True)  # type: ignore

    model_names = set(
        [
            m.model_name
            for group in cached_samples(eval_dir, None)
            for m in group.model_evals
        ]
    )
    with st.form("comp"):
        st.subheader("Model Comparison Tool")
        base_model = st.selectbox("Base model", model_names)
        compare_model = st.selectbox("Compare model", model_names)
        st.text(f"Comparing {base_model} with {compare_model}")
        submitted = st.form_submit_button("Compare")
        if base_model and compare_model and submitted:
            print("Computing comparisons")
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
