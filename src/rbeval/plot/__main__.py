import argparse
from pathlib import Path
from typing import Callable, List, Optional
from rbeval.plot.data import EvalGroup, Figure, get_samples
from rbeval.plot.model_comp import model_comparer
from tqdm import tqdm
from rbeval.plot.score_cdf import score_cdf

plot_fns: List[Callable[[List[EvalGroup], List[str]], List[Figure]]] = [
    score_cdf,
    model_comparer,
]


def main():
    parser = argparse.ArgumentParser(description="Generate performance curves")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("figure_dir", type=str)
    parser.add_argument("-n", "--name", type=str)
    args, rest = parser.parse_known_args()

    name_filter: Optional[str] = args.name
    eval_dir = Path(args.eval_dir)
    figure_dir = Path(args.figure_dir)
    samples = get_samples(eval_dir, name_filter)

    figures = [fig for fig_fn in plot_fns for fig in fig_fn(samples, rest)]
    for figure in (pbar := tqdm(figures, desc="Saving figures")):
        path = (figure_dir / figure.name.replace(".", "_")).with_suffix(".png")
        pbar.set_description(f"Saving {path.stem}")
        figure.chart.save(str(path), scale_factor=2.0)  # type: ignore


if __name__ == "__main__":
    main()
