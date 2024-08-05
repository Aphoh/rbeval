import subprocess
import argparse
from typing import Optional
import torch
import warnings
import os
from pathlib import Path

from rbeval.eval_spec import EvalSpec, rand_uid


def run_lm_eval(
    lm_eval_path: Optional[str],
    model_args: str,
    tasks: str,
    num_fewshot: int,
    output_path: str,
):
    if lm_eval_path is None:
        cmd = ["python3", "-m", "lm_eval"]
    else:
        cmd = [lm_eval_path]
    cmd += ["--model", "vllm"]
    cmd += ["--model_args", model_args]
    cmd += ["--tasks", tasks]
    cmd += ["--num_fewshot", str(num_fewshot)]
    cmd += ["--output_path", output_path]
    cmd += ["--log_samples"]
    cmd += ["--cache_requests", "true"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=os.environ)


def main():
    parser = argparse.ArgumentParser(description="Run eval for a given model")

    parser.add_argument("output", type=str, help="output directory")
    parser.add_argument("model", type=str, help="model path")
    parser.add_argument("group", type=str)

    parser.add_argument("--lm_eval_path", type=str)
    parser.add_argument("--req_cache_path", type=str, default="/tmp/lm_eval_cache")
    parser.add_argument("--tasks", type=str, default="mmlu")
    parser.add_argument("--min_fewshot", type=int, default=0)
    parser.add_argument("--max_fewshot", type=int, default=0)
    parser.add_argument("-r", "--reformat", type=str)

    args = parser.parse_args()
    model: str = args.model
    output_path: Path = Path(args.output)
    group: str = args.group
    reformat: Optional[str] = args.reformat
    lm_eval_path: Optional[str] = args.lm_eval_path
    req_cache_path = Path(args.req_cache_path)
    tasks: str = args.tasks
    min_fewshot: int = args.min_fewshot
    max_fewshot: int = args.max_fewshot
    max_fewshot = max(min_fewshot, max_fewshot)

    if not output_path.exists():
        warnings.warn(f"Output path {output_path} does not exist, creating it")
        output_path.mkdir(parents=True)
    if not req_cache_path.exists():
        warnings.warn(f"Request cache path, {str(req_cache_path)} does not exist")

    os.environ["LM_HARNESS_CACHE_PATH"] = args.req_cache_path

    n_gpu = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(n_gpu)))
    model_args = f"pretrained={model},dtype=auto,gpu_memory_utilization=0.7,tensor_parallel_size=1,data_parallel_size={n_gpu},max_model_len=4096"
    fewshot = list(range(min_fewshot, max_fewshot + 1))

    for num_fewshot in fewshot:
        cfg = EvalSpec(
            uid=rand_uid(),
            model=model,
            model_name=model.split("/")[-1],
            group=group,
            model_args=model_args,
            fewshot=num_fewshot,
            tasks=tasks,
        )
        spec_name = cfg.name()
        lm_eval_output_path = output_path / spec_name
        if reformat:
            ref_path = Path(reformat)
            assert ref_path.exists()
            ref_files = list(ref_path.glob("**/*.json*"))
            print(f"Found dir to reformat with {len(ref_files)} files")
            assert len(ref_files) > 0
            lm_eval_output_path.mkdir(parents=True, exist_ok=False)
            for file in ref_files:
                file.rename(lm_eval_output_path / file.name)
        else:
            run_lm_eval(
                lm_eval_path, model_args, tasks, num_fewshot, str(lm_eval_output_path)
            )

        # Succeeded, write config
        cfg_path = output_path / f"{spec_name}.json"
        with open(cfg_path, "w") as f:
            f.write(cfg.json())


if __name__ == "__main__":
    main()
