from dataclasses import dataclass, field
from functools import cached_property
import re
from typing import Dict, List
from collections import defaultdict

import numpy as np

from rbeval.eval_spec import EvalSpec


@dataclass
class Eval:
    name: str
    cor_logprobs: np.ndarray
    """shape [n] array of correct logprobs"""
    inc_logprobs: np.ndarray
    """shape [n, k] array of incorrect logprobs"""

    def task_name(self) -> str:
        return self.name.rsplit("_", 1)[0]

    def filter_inds(self, where: np.ndarray) -> "Eval":
        return Eval(
            name=self.name,
            cor_logprobs=self.cor_logprobs[where],
            inc_logprobs=self.inc_logprobs[where],
        )

    def filter_mask(self, mask: np.ndarray) -> "Eval":
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

    def model_label(self, model_name) -> str:
        return f"{model_name} fs{self.min_fewshots[model_name]}-{self.max_fewshots[model_name]}"

    def collect_with_name(self, name_filt) -> List[ModelEval]:
        """Collects all evals that contain the model_name"""
        return [
            m for m in self.model_evals if re.match(name_filt, m.model_name) is not None
        ]
