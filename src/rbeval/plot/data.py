from dataclasses import dataclass, field
from typing import List

import numpy as np

from rbeval.eval_spec import EvalSpec


@dataclass
class Eval:
    name: str
    cor_logprobs: np.ndarray
    """shape [n] array of correct logprobs"""
    inc_logprobs: np.ndarray
    """shape [n, k] array of incorrect logprobs"""


@dataclass
class ModelEval:
    """The evaluations for a given model"""

    model_spec: EvalSpec
    evals: List[Eval] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model_spec.model_name


@dataclass
class EvalGroup:
    """A group of model evals"""

    group: str
    model_evals: List[ModelEval] = field(default_factory=list)
