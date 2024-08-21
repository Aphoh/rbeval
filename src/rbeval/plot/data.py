from dataclasses import dataclass, field, asdict
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Dict, List, Optional
from collections import defaultdict
import altair as alt

from dacite import from_dict
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

from rbeval.eval_spec import EvalSpec


def get_samples(inp: Path, name_filter: Optional[str]) -> List["EvalGroup"]:
    groups: Dict[str, EvalGroup] = {}

    for spec_file in (pbar := tqdm(list(inp.glob("*.json")), desc="Reading specs")):
        pbar.set_description(f"Reading spec {spec_file.stem}")
        with open(spec_file) as f:
            spec = EvalSpec(**json.load(f))

        if name_filter:
            if re.match(name_filter, spec.model_name) is None:
                print(f"Skipping spec {spec_file.stem}")
                continue

        group = groups.setdefault(spec.group, EvalGroup(name=spec.group))
        model_eval_cache_file = Path(
            spec_file.with_stem(spec_file.stem + "_cache")
        ).with_suffix(".npy")
        if model_eval_cache_file.exists():
            res_dict = np.load(str(model_eval_cache_file), allow_pickle=True).item()
            model_eval = from_dict(data_class=ModelEval, data=res_dict)
            groups[group.name].model_evals.append(model_eval)
            continue
        else:
            model_eval = ModelEval(eval_spec=spec)
            group.model_evals.append(model_eval)
            for samples_file in (spec_file.parent / spec_file.stem).glob(
                "**/samples_*.json*"
            ):
                with open(samples_file, "r") as f:
                    if samples_file.suffix == ".jsonl":
                        docs = [json.loads(s) for s in f.readlines()]
                    else:
                        assert samples_file.suffix == ".json"
                        docs = json.load(f)

                cor_logprobs: List[float] = []
                inc_logprobs: List[List[float]] = []
                for doc in docs:
                    target = doc["target"]
                    probs: List[float] = [float(a[0][0]) for a in doc["resps"]]
                    cor_logprobs.append(probs.pop(target))
                    inc_logprobs.append(probs)
                eval = Eval(
                    name=samples_file.stem,
                    cor_logprobs=np.array(cor_logprobs, dtype=np.float64),
                    inc_logprobs=np.array(inc_logprobs, dtype=np.float64),
                )
                model_eval.evals.append(eval)
            np.save(str(model_eval_cache_file), asdict(model_eval))  # type: ignore

    return list(groups.values())


@dataclass
class Eval:
    name: str
    cor_logprobs: np.ndarray
    """shape [n] array of correct logprobs"""
    inc_logprobs: np.ndarray
    """shape [n, k] array of incorrect logprobs"""

    def task_name(self) -> str:
        return self.name.rsplit("_", 1)[0]

    def filter_inds(self, where: NDArray[np.int64]) -> "Eval":
        return Eval(
            name=self.name,
            cor_logprobs=self.cor_logprobs[where],
            inc_logprobs=self.inc_logprobs[where],
        )

    def filter_mask(self, mask: NDArray[np.bool_]) -> "Eval":
        return self.filter_inds(np.where(mask)[0])


@dataclass
class ModelEval:
    """The evaluations for a given model"""

    eval_spec: EvalSpec
    evals: List[Eval] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.eval_spec.model_name

    def n_samples(self) -> int:
        return sum(len(e.cor_logprobs) for e in self.evals)


@dataclass
class EvalGroup:
    """A group of model evals"""

    name: str
    model_evals: List[ModelEval] = field(default_factory=list)

    def model_names(self) -> List[str]:
        return [m.model_name for m in self.model_evals]

    @cached_property
    def min_fewshots(self) -> Dict[str, int]:
        res: Dict[str, int] = defaultdict(lambda: 999)
        for me in self.model_evals:
            res[me.model_name] = min(res[me.model_name], me.eval_spec.fewshot)
        return res

    @cached_property
    def max_fewshots(self) -> Dict[str, int]:
        res: Dict[str, int] = defaultdict(lambda: -999)
        for me in self.model_evals:
            res[me.model_name] = max(res[me.model_name], me.eval_spec.fewshot)
        return res

    def model_label(self, model_name: str) -> str:
        return f"{model_name} fs{self.min_fewshots[model_name]}-{self.max_fewshots[model_name]}"

    def collect_with_name(self, name_filt: str) -> List[ModelEval]:
        """Collects all evals that contain the model_name"""
        return [
            m for m in self.model_evals if re.match(name_filt, m.model_name) is not None
        ]


@dataclass
class Figure:
    """A figure to plot"""

    name: str
    chart: (
        alt.Chart
        | alt.LayerChart
        | alt.HConcatChart
        | alt.ConcatChart
        | alt.VConcatChart
    )
    group: Optional[str] = None
