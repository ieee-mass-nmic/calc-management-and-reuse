"""Normalize raw task exports without discarding evaluation metadata."""
from __future__ import annotations

import json
from pathlib import Path


def normalize_record(record: dict, task: str) -> dict:
    if "instruction" in record and "response" in record:
        return record
    meta = dict(record.get("meta", {}))
    if task == "gsm8k":
        instruction, response = record["question"], record["answer"]
    elif task == "mmlu":
        choices = record["choices"]
        if len(choices) != 4:
            raise ValueError("MMLU expects four choices")
        instruction = record["question"] + "\n" + "\n".join(
            f"{label}. {choice}" for label, choice in zip("ABCD", choices)) + "\nAnswer:"
        answer = record["answer"]
        response = "ABCD"[answer] if isinstance(answer, int) else str(answer).strip()
        if response not in list("ABCD"):
            raise ValueError("invalid MMLU answer")
        meta["choices"] = choices
    elif task == "tydiqa":
        answers = record["answers"]
        answers = answers["text"] if isinstance(answers, dict) else answers
        if not answers or not all(isinstance(x, str) for x in answers):
            raise ValueError("TyDiQA requires nonempty text answers (GoldP schema)")
        instruction = record["context"] + "\nQuestion: " + record["question"] + "\nAnswer:"
        response = answers[0]
        meta["answers"] = answers
    elif task == "humaneval":
        instruction, response = record["prompt"], record["canonical_solution"]
        meta.update({k: record[k] for k in ("test", "entry_point")})
    elif task == "pool":
        instruction = record["instruction"]
        if record.get("input"):
            instruction += "\n" + record["input"]
        response = record["output"]
    else:
        raise ValueError(f"unsupported raw task schema: {task}")
    return {"instruction": instruction, "response": response, "meta": meta,
            "source": record.get("source"), "language": record.get("language")}


def prepare_jsonl(source: Path, destination: Path, task: str) -> Path:
    from pcu_select.data import JsonlPool

    rows = [normalize_record(json.loads(line), task) for line in source.read_text().splitlines() if line.strip()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".jsonl.tmp")
    try:
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        JsonlPool.from_jsonl(temp)  # validate duplicates and required types before publication
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination
