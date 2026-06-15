"""
Методы адаптации без дообучения. Каждый возвращает (текст_ответа, число_токенов).
Ни градиентов, ни обновления весов — только инференс.

  zero_shot          — прямой ответ без примеров
  few_shot           — k случайных демонстраций в контексте
  cot                — zero-shot chain-of-thought
  few_shot_cot       — k демонстраций с цепочками рассуждений
  self_consistency   — N цепочек CoT, ответ по большинству (стоимость = сумма всех сэмплов)
  knn_few_shot_cot   — демонстрации по близости эмбеддингов к запросу
"""

from collections import Counter

from .prompts import build_gsm8k, build_mmlu
from .tasks import extract_number, extract_letter


def _build(task, item, demos, cot):
    if task == "gsm8k":
        return build_gsm8k(item["question"], demos, cot)
    return build_mmlu(item, demos, cot)


def _extract(task, text):
    return extract_number(text) if task == "gsm8k" else extract_letter(text)


def zero_shot(llm, task, item):
    out = llm.chat(_build(task, item, [], cot=False), max_new_tokens=16)
    return out, llm.count_tokens(out)


def few_shot(llm, task, item, demos):
    out = llm.chat(_build(task, item, demos, cot=False), max_new_tokens=16)
    return out, llm.count_tokens(out)


def cot(llm, task, item):
    out = llm.chat(_build(task, item, [], cot=True), max_new_tokens=512)
    return out, llm.count_tokens(out)


def few_shot_cot(llm, task, item, demos):
    out = llm.chat(_build(task, item, demos, cot=True), max_new_tokens=512)
    return out, llm.count_tokens(out)


def self_consistency(llm, task, item, n_samples=5, temperature=0.7):
    msgs = _build(task, item, [], cot=True)
    votes, total_tokens = [], 0
    for _ in range(n_samples):
        out = llm.chat(msgs, max_new_tokens=512, do_sample=True, temperature=temperature)
        total_tokens += llm.count_tokens(out)
        ans = _extract(task, out)
        if ans is not None:
            votes.append(ans)
    if not votes:
        return "", total_tokens
    winner = Counter(votes).most_common(1)[0][0]
    return f"The answer is {winner}", total_tokens


def knn_few_shot_cot(llm, task, item, retriever, k=3):
    demos = retriever.topk(item["question"], k=k)
    out = llm.chat(_build(task, item, demos, cot=True), max_new_tokens=512)
    return out, llm.count_tokens(out)
