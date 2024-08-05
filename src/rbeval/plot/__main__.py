import argparse
from pathlib import Path
import json
from typing import Dict, Optional
import numpy as np
from dataclasses import asdict
import re
from rbeval.eval_spec import EvalSpec
from rbeval.plot.data import Eval, EvalGroup, ModelEval
from rbeval.plot.score_cdf import score_cdf
from tqdm import tqdm

plot_fns = [score_cdf]


def get_samples(inp: Path, name_filter: Optional[str]) -> Dict[str, EvalGroup]:
    groups: Dict[str, EvalGroup] = {}

    for spec_file in (pbar := tqdm(list(inp.glob("*.json")), desc="Reading specs")):
        pbar.set_description(f"Reading spec {spec_file.stem}")
        with open(spec_file) as f:
            spec = EvalSpec(**json.load(f))

        if name_filter:
            if re.match(name_filter, spec.model_name) is None:
                print(f"Skipping spec {spec_file.stem}")
                continue

        group = groups.setdefault(spec.group, EvalGroup(group=spec.group))
        model_eval = ModelEval(eval_spec=spec)
        group.model_evals.append(model_eval)
        for samples_file in (spec_file.parent / spec_file.stem).glob(
            "**/samples_*.json*"
        ):
            cache_file = samples_file.with_suffix(".npy")
            if samples_file.with_suffix(".npy").exists():
                model_eval.evals.append(
                    Eval(**np.load(str(cache_file), allow_pickle=True).item())
                )
            else:
                with open(samples_file, "r") as f:
                    if samples_file.suffix == ".jsonl":
                        docs = [json.loads(s) for s in f.readlines()]
                    else:
                        assert samples_file.suffix == ".json"
                        docs = json.load(f)

                cor_logprobs = []
                inc_logprobs = []
                for doc in docs:
                    target = doc["target"]
                    probs = [float(a[0][0]) for a in doc["resps"]]
                    cor_logprobs.append(probs.pop(target))
                    inc_logprobs.append(probs)
                eval = Eval(
                    name=samples_file.stem,
                    cor_logprobs=np.array(cor_logprobs),
                    inc_logprobs=np.array(inc_logprobs),
                )
                np.save(str(cache_file), asdict(eval))  # type: ignore
                model_eval.evals.append(eval)

    return groups


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

    for fn in plot_fns:
        fn(samples, figure_dir, rest)


if __name__ == "__main__":
    main()
