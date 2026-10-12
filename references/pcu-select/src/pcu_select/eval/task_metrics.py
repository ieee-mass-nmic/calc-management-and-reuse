"""Repository-owned downstream protocols (scores in [0, 100]).

GSM8K: greedy numeric exact match; MMLU: conditional likelihood of A/B/C/D;
TyDiQA: best-reference Unicode-normalized token F1; HumanEval: greedy pass@1.
HumanEval executes generated code only in a restricted, disposable Docker container.
These explicit protocols are not a claim of equivalence to every leaderboard recipe.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import re
import shutil
import subprocess
import unicodedata
from typing import Any

import numpy as np
import torch

from pcu_select.features.tokenization import encode_response_lm
from pcu_select.types import ValidationSketch


def numeric_answer(text: str) -> Decimal | None:
    text = text.rsplit("####", 1)[-1].replace(",", "")
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    try:
        return Decimal(values[-1]) if values else None
    except InvalidOperation:
        return None


def answer_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, truth = answer_tokens(prediction), answer_tokens(reference)
    if not pred or not truth:
        return float(pred == truth)
    common = sum((Counter(pred) & Counter(truth)).values())
    return 2 * common / (len(pred) + len(truth))


def execute_humaneval(code: str, *, image: str, timeout: float) -> bool:
    """No host mounts, network, inherited credentials, or privileged execution."""
    import uuid

    name = "pcu-eval-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network=none",
               "--read-only", "--memory=512m", "--cpus=1", "--pids-limit=32",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
               "--tmpfs=/tmp:rw,noexec,nosuid,size=32m", "-i", image, "python", "-I", "-S", "-"]
    try:
        result = subprocess.run(command, input=code, text=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=timeout)
        if result.returncode in (125, 126, 127):
            raise RuntimeError("HumanEval Docker runtime failed; this is not a failed solution")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=10, check=False)


@dataclass
class BuiltinTaskMetric:
    task_name: str
    eval_set: ValidationSketch
    max_new_tokens: int = 256
    max_len: int = 1024
    code_image: str = "python:3.11-slim"
    code_timeout: float = 10.0
    predictions: list[dict] = field(default_factory=list)

    @property
    def metric_name(self) -> str:
        return {"gsm8k": "exact_match", "mmlu": "accuracy", "tydiqa": "f1",
                "humaneval": "pass@1"}[self.task_name]

    def validate(self) -> None:
        if self.task_name not in ("gsm8k", "mmlu", "tydiqa", "humaneval"):
            raise ValueError(f"no builtin metric for {self.task_name}; supply a metric factory")
        if not self.eval_set.samples or self.max_new_tokens <= 0:
            raise ValueError("evaluation requires samples and a positive generation budget")
        for sample in self.eval_set.samples:
            if self.task_name == "gsm8k" and numeric_answer(sample.response) is None:
                raise ValueError(f"GSM8K reference has no numeric answer: {sample.sample_id}")
            if self.task_name == "mmlu" and sample.response.strip() not in "ABCD":
                raise ValueError("MMLU response must be one of A/B/C/D; use prepare_data.py")
            if self.task_name == "humaneval" and not all(
                sample.meta.get(k) for k in ("test", "entry_point")
            ):
                raise ValueError("HumanEval requires meta.test and meta.entry_point")
        if self.task_name == "humaneval":
            if shutil.which("docker") is None:
                raise RuntimeError("HumanEval requires Docker for isolated code evaluation")
            check = subprocess.run(["docker", "image", "inspect", self.code_image],
                                   capture_output=True, timeout=15)
            if check.returncode:
                raise RuntimeError(f"prepare the HumanEval container first: docker pull {self.code_image}")

    @torch.no_grad()
    def __call__(self, model: Any, tokenizer: Any) -> tuple[str, float]:
        self.validate()
        model.eval()
        device = next(model.parameters()).device
        self.predictions = []
        scores = []
        for sample in self.eval_set.samples:
            if self.task_name == "mmlu":
                likelihoods = []
                for choice in "ABCD":
                    enc = encode_response_lm(tokenizer, sample.instruction, choice, max_len=self.max_len)
                    ids = torch.tensor([enc["input_ids"]], device=device)
                    logits = model(input_ids=ids).logits[0, :-1].float().log_softmax(-1)
                    mask = torch.tensor(enc["response_mask"][1:], device=device).bool()
                    # Compare the label itself, not EOS likelihood.
                    if ids[0, -1].item() == tokenizer.eos_token_id:
                        mask[-1] = False
                    if not mask.any():
                        raise ValueError("MMLU answer has no scored tokens")
                    ll = logits.gather(1, ids[0, 1:, None]).squeeze(1)[mask].sum()
                    likelihoods.append(float(ll))
                prediction = "ABCD"[int(np.argmax(likelihoods))]
                score = float(prediction == sample.response.strip())
            else:
                tokens = tokenizer(sample.instruction, add_special_tokens=True)["input_ids"]
                context = int(getattr(model.config, "max_position_embeddings", self.max_len + self.max_new_tokens))
                keep = min(self.max_len, context - self.max_new_tokens)
                if keep <= 0:
                    raise ValueError("generation budget exceeds model context window")
                ids = torch.tensor([tokens[-keep:]], device=device)
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                prediction = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
                if self.task_name == "gsm8k":
                    answer = numeric_answer(prediction)
                    score = float(answer is not None and answer == numeric_answer(sample.response))
                elif self.task_name == "tydiqa":
                    references = sample.meta.get("answers", [sample.response])
                    score = max(token_f1(prediction, ref) for ref in references)
                else:
                    code = sample.instruction + prediction + "\n" + sample.meta["test"]
                    entry = sample.meta["entry_point"]
                    if not re.fullmatch(r"[A-Za-z_]\w*", entry):
                        raise ValueError("invalid HumanEval entry point")
                    code += f"\ncheck({entry})\n"
                    score = float(execute_humaneval(code, image=self.code_image, timeout=self.code_timeout))
            scores.append(score)
            self.predictions.append({"sample_id": sample.sample_id, "prediction": prediction,
                                     "reference": sample.response, "score": score})
        return self.metric_name, 100.0 * float(np.mean(scores))

    def save_predictions(self, path) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.predictions))


def build_task_metric(*, task_name: str, eval_set: ValidationSketch, args=None, **_) -> BuiltinTaskMetric:
    metric = BuiltinTaskMetric(task_name, eval_set,
        max_new_tokens=getattr(args, "max_new_tokens", 256),
        max_len=getattr(args, "max_len", 1024),
        code_image=getattr(args, "code_image", "python:3.11-slim"))
    metric.validate()
    return metric
