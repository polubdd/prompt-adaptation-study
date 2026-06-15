"""Загрузка GSM8K (рассуждение) и подмножества MMLU (знания) + извлечение ответов."""

import re
import pandas as pd
from huggingface_hub import hf_hub_download


MMLU_SUBJECTS = [
    "high_school_world_history",
    "miscellaneous",
    "marketing",
    "nutrition",
    "professional_psychology",
]
LETTERS = ["A", "B", "C", "D"]


def _read_parquet(repo: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    return pd.read_parquet(path)


def load_gsm8k(n_test: int = 200, n_pool: int = 100, seed: int = 0):
    train = (_read_parquet("openai/gsm8k", "main/train-00000-of-00001.parquet")
             .sample(frac=1, random_state=seed).reset_index(drop=True))
    test = (_read_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet")
            .sample(frac=1, random_state=seed).reset_index(drop=True))

    def parse(row):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        cot = row["answer"].split("####")[0].strip()
        return {"question": row["question"], "answer": int(gold), "cot": cot}

    test_items = [parse(r) for _, r in test.head(n_test).iterrows()]
    pool_items = [parse(r) for _, r in train.head(n_pool).iterrows()]
    return test_items, pool_items


def load_mmlu(n_test_per_subject: int = 30, n_pool: int = 50, seed: int = 0):
    test_items, pool_items = [], []
    for subj in MMLU_SUBJECTS:
        test = (_read_parquet("cais/mmlu", f"{subj}/test-00000-of-00001.parquet")
                .sample(frac=1, random_state=seed).reset_index(drop=True))
        dev = _read_parquet("cais/mmlu", f"{subj}/dev-00000-of-00001.parquet")
        for _, r in test.head(n_test_per_subject).iterrows():
            test_items.append({
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": LETTERS[int(r["answer"])],
                "subject": subj,
            })
        for _, r in dev.iterrows():
            pool_items.append({
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": LETTERS[int(r["answer"])],
                "subject": subj,
            })
    return test_items, pool_items[:n_pool]


_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_number(text: str):
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace("$", "").replace(",", "")
    try:
        val = float(raw)
        return int(val) if val.is_integer() else val
    except ValueError:
        return None


def extract_letter(text: str):
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def is_correct(task: str, prediction: str, gold) -> bool:
    if task == "gsm8k":
        pred = extract_number(prediction)
        return pred is not None and pred == gold
    if task == "mmlu":
        return extract_letter(prediction) == gold
    raise ValueError(task)
