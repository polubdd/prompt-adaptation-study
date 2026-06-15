"""
Полный прогон: все методы адаптации на GSM8K и MMLU. Рассчитано на Colab T4.
Обучения нет нигде.

    python run_all.py            полный прогон
    python run_all.py --quick    быстрый смоук-тест на мелкой выборке
"""

import argparse

from src.model import FrozenLLM
from src.tasks import load_gsm8k, load_mmlu
from src.retriever import DemoRetriever
from src.evaluate import evaluate_method, save_results
from src.plot_results import (
    load_data, plot_accuracy_by_method, plot_accuracy_vs_cost,
    plot_kshot_curve, generate_markdown_table,
)

METHODS = ["zero_shot", "few_shot", "cot", "few_shot_cot",
           "knn_few_shot_cot", "self_consistency"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    n_test = 20 if args.quick else 200
    n_mmlu = 5 if args.quick else 30

    print("Loading model (frozen, no training)...")
    llm = FrozenLLM()

    print("Loading tasks...")
    gsm_test, gsm_pool = load_gsm8k(n_test=n_test)
    mmlu_test, mmlu_pool = load_mmlu(n_test_per_subject=n_mmlu)

    gsm_retr = DemoRetriever(gsm_pool)
    mmlu_retr = DemoRetriever(mmlu_pool)

    results = []
    for task, test, pool, retr in [
        ("gsm8k", gsm_test, gsm_pool, gsm_retr),
        ("mmlu",  mmlu_test, mmlu_pool, mmlu_retr),
    ]:
        for m in METHODS:
            r = evaluate_method(llm, task, test, m, demos_pool=pool,
                                retriever=retr, k=args.k, n_samples=args.n_samples)
            results.append(r)
            print(f"  {task}/{m}: acc={r['accuracy']}%  tokens={r['avg_completion_tokens']}")

    save_results(results)

    df = load_data()
    print("\n" + generate_markdown_table(df))
    plot_accuracy_by_method(df)
    plot_accuracy_vs_cost(df)
    plot_kshot_curve()
    print("\nDone! See results/ and results/figures/.")


if __name__ == "__main__":
    main()
