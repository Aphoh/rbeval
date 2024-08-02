from dataclasses import dataclass, asdict
import json
import uuid


def rand_uid():
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class EvalSpec:
    uid: str
    model: str
    model_name: str
    group: str
    model_args: str
    fewshot: int
    tasks: str

    def json(self) -> str:
        return json.dumps(asdict(self))

    def name(self) -> str:
        return (
            f"{self.group}_{self.model_name}_fs{self.fewshot}_{self.tasks}_{self.uid}"
        )

    def pretty_name(self) -> str:
        return f"{self.model_name} fs{self.fewshot}"
