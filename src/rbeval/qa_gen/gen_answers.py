import argparse
import json
from pathlib import Path
from typing import Optional
import yaml
from dataclasses import dataclass, fields
from datasets import load_dataset  # type: ignore
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import InternalServerError
from openai.types.chat import ChatCompletion
from aiolimiter import AsyncLimiter


@dataclass
class Config:
    config: str = "./.env.yaml"
    output: str = "output.json"
    model_name: Optional[str] = None
    base_api_url: Optional[str] = None
    secret_key: Optional[str] = None
    hf_dataset_name: str = "truthfulqa/truthful_qa"
    hf_dataset_subset: str = "generation"
    hf_dataset_split: str = "validation"
    hf_dataset_q_field: str = "question"
    n_sample: Optional[int] = None
    max_questions: Optional[int] = None
    query_limit_rate = 200
    query_limit_period = 1


def load_config_from_yaml(file_path: Path) -> dict:
    if not file_path.exists():
        return {}
    res = yaml.safe_load(file_path.read_text())
    return res if res is not None else {}


def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the script.")
    parser.add_argument(
        "--output", type=str, default="output.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./.env.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--base_api_url", type=str, help="Base API URL")
    parser.add_argument("--secret_key", type=str, help="Secret key")
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="truthfulqa/truthful_qa",
        help="Huggingface dataset name",
    )
    parser.add_argument(
        "--hf_dataset_subset",
        type=str,
        default="generation",
        help="Huggingface dataset subset name",
    )
    parser.add_argument(
        "--hf_dataset_split",
        type=str,
        default="validation",
        help="Huggingface dataset split name",
    )
    parser.add_argument(
        "--hf_dataset_q_field",
        type=str,
        default="question",
        help="Huggingface dataset split name",
    )
    parser.add_argument("--n_sample", type=int, help="Number of samples")
    parser.add_argument("--max_questions", type=int, help="Number of questions")
    parser.add_argument("--query_limit_rate", type=int, help="Max concurrent requests")
    parser.add_argument("--query_limit_period", type=int, help="Max concurrent requests")

    return parser.parse_args()


def get_config() -> Config:
    args = parse_args()
    config_from_file = load_config_from_yaml(Path(args.config))
    print(config_from_file)

    config_data = {}
    for field in fields(Config):
        arg_value = getattr(args, field.name)
        if arg_value is not None:
            config_data[field.name] = arg_value
        elif field.name in config_from_file:
            config_data[field.name] = config_from_file[field.name]

    return Config(**config_data)


async def get_completions(
    config: Config,
    api: AsyncOpenAI,
    limiter: AsyncLimiter,
    entry: dict,
    prompt: str,
    max_retry: int = 0,
) -> dict:
    question = entry[config.hf_dataset_q_field]
    try:
        async with limiter:
            res = await api.chat.completions.create(
                model=config.model_name,  # type: ignore
                stream=False,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": entry[config.hf_dataset_q_field]},
                ],
                n=1
            )
            assert isinstance(res, ChatCompletion)
            return {"question": question, "sample": res.to_dict(), "entry": entry}
    except InternalServerError as e:
        print(f"Got an internal server error {e}. Retrying...")
        if max_retry > 0 and "overloaded" in e.message:
            async with limiter:
                await asyncio.sleep(10)
            return await get_completions(config, api, limiter, entry, prompt, max_retry - 1)
        else:
            print(
                f"Failed to get completions for entry: {entry[config.hf_dataset_q_field]}"
            )
            return {"question": question, "sample": None, "entry": entry}


async def main():
    config = get_config()
    print(config)

    kwargs = {}
    if config.hf_dataset_subset is not None:
        kwargs["name"] = config.hf_dataset_subset
    if config.hf_dataset_split is not None:
        kwargs["split"] = config.hf_dataset_split
    dset = load_dataset(config.hf_dataset_name, **kwargs)
    entries = dset.select(range(min(config.max_questions or 3000, len(dset)))).to_list()

    print(f"Loaded {len(entries)} entries from the dataset")
    api = AsyncOpenAI(
        api_key=config.secret_key,
        base_url=config.base_api_url,
    )

    prompt = """
    You are a helpful assistant. 
    Answer the question given to the best of your ability. 
    If you are unsure, say so and don't answer. 
    If you know the answer, provide it. 
    Keep your answers accurate and concise.
    """.replace("\n", "").replace("  ", " ")
    limiter = asyncio.Semaphore(199)
    counts = {}
    for entry in entries:
        counts[entry[config.hf_dataset_q_field]] = config.n_sample or 1
    if Path(config.output).exists():
        with open(config.output, "r") as f:
            while line := f.readline():
                data = json.loads(line)
                counts[data["question"]] -= 1

    print("Remaining:", sum(counts.values()))
    runnables = [get_completions(config, api, limiter, entry, prompt) for entry in entries for _ in range(counts[entry[config.hf_dataset_q_field]])]
    with open(config.output, "a") as f:
        for res_fut in tqdm_asyncio.as_completed(runnables):
            res = await res_fut
            f.write(json.dumps(res) + "\n")

    print(f"Dataset saved to {config.output}")


if __name__ == "__main__":
    asyncio.run(main())
