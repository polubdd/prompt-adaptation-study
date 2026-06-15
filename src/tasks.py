"""Загрузка GSM8K (рассуждение) и подмножества MMLU (знания) + извлечение ответов."""

import re
from datasets import load_dataset


MMLU_SUBJECTS = [
    "high_school_world_history",
    "miscellaneous",
    "marketing",
    "nutrition",
    "professional_psychology",
]
LETTERS = ["A", "B", "C", "D"]


def load_gsm8k(n_test: int = 200, n_pool: int = 100, seed: int = 0):
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)

    def parse(row):
        sol = row["answer"]
        gold = sol.split("####")[-1].strip().replace(",", "")
        cot = sol.split("####")[0].strip()
        return {"question": row["question"], "answer": int(gold), "cot": cot}

    test_items = [parse(r) for r in test.select(range(n_test))]
    pool_items = [parse(r) for r in train.select(range(n_pool))]
    return test_items, pool_items


def load_mmlu(n_test_per_subject: int = 30, n_pool: int = 50, seed: int = 0):
    test_items, pool_items = [], []
    for subj in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj)
        test = ds["test"].shuffle(seed=seed)
        for r in test.select(range(min(n_test_per_subject, len(test)))):
            test_items.append({
                "question": r["question"],
                "choices": r["choices"],
                "answer": LETTERS[r["answer"]],
                "subject": subj,
            })
        dev = ds["dev"] if "dev" in ds else ds["validation"]
        for r in dev:
            pool_items.append({
                "question": r["question"],
                "choices": r["choices"],
                "answer": LETTERS[r["answer"]],
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
