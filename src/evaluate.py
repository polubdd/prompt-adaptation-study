"""Прогон метода по задаче: точность + средняя стоимость ответа в токенах."""

import json
import random
from pathlib import Path
from tqdm import tqdm

from . import methods
from .tasks import is_correct


def evaluate_method(llm, task, test_items, method_name, demos_pool=None,
                    retriever=None, k=3, n_samples=5, seed=0):
    rng = random.Random(seed)
    correct, token_counts = 0, []

    for item in tqdm(test_items, desc=f"{task}/{method_name}", leave=False):
        if method_name == "zero_shot":
            out, tok = methods.zero_shot(llm, task, item)
        elif method_name.startswith("few_shot_cot"):
            demos = rng.sample(demos_pool, k)
            out, tok = methods.few_shot_cot(llm, task, item, demos)
        elif method_name.startswith("few_shot"):
            demos = rng.sample(demos_pool, k)
            out, tok = methods.few_shot(llm, task, item, demos)
        elif method_name == "cot":
            out, tok = methods.cot(llm, task, item)
        elif method_name == "self_consistency":
            out, tok = methods.self_consistency(llm, task, item, n_samples=n_samples)
        elif method_name == "knn_few_shot_cot":
            out, tok = methods.knn_few_shot_cot(llm, task, item, retriever, k=k)
        else:
            raise ValueError(method_name)

        token_counts.append(tok)
        if is_correct(task, out, item["answer"]):
            correct += 1

    avg_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else 0
    return {
        "method": method_name,
        "task": task,
        "accuracy": round(100 * correct / len(test_items), 2),
        "n": len(test_items),
        "avg_completion_tokens": avg_tokens,
    }


def save_results(results: list, out_path: str = "./results/results.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
