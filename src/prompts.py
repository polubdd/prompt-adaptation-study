"""Сборка промптов (messages для chat-шаблона) под каждый метод адаптации."""


GSM8K_SYS_DIRECT = (
    "You are a helpful assistant. Answer the math question. "
    "Give ONLY the final numeric answer, nothing else."
)
GSM8K_SYS_COT = (
    "You are a helpful assistant. Solve the math question step by step, "
    "then give the final answer on a new line as 'The answer is <number>'."
)


def _gsm8k_demo(item, cot: bool):
    if cot:
        assistant = f"{item['cot']}\nThe answer is {item['answer']}"
    else:
        assistant = str(item["answer"])
    return [{"role": "user", "content": item["question"]},
            {"role": "assistant", "content": assistant}]


def build_gsm8k(question: str, demos: list, cot: bool):
    messages = [{"role": "system", "content": GSM8K_SYS_COT if cot else GSM8K_SYS_DIRECT}]
    for d in demos:
        messages += _gsm8k_demo(d, cot)
    messages.append({"role": "user", "content": question})
    return messages


MMLU_SYS_DIRECT = (
    "You are a helpful assistant. Answer the multiple-choice question. "
    "Respond with ONLY the letter (A, B, C, or D)."
)
MMLU_SYS_COT = (
    "You are a helpful assistant. Reason briefly about the multiple-choice "
    "question, then end with 'The answer is <letter>'."
)


def _format_mmlu_question(item) -> str:
    lines = [item["question"]]
    for letter, choice in zip("ABCD", item["choices"]):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def _mmlu_demo(item, cot: bool):
    assistant = f"The answer is {item['answer']}" if cot else item["answer"]
    return [{"role": "user", "content": _format_mmlu_question(item)},
            {"role": "assistant", "content": assistant}]


def build_mmlu(item, demos: list, cot: bool):
    messages = [{"role": "system", "content": MMLU_SYS_COT if cot else MMLU_SYS_DIRECT}]
    for d in demos:
        messages += _mmlu_demo(d, cot)
    messages.append({"role": "user", "content": _format_mmlu_question(item)})
    return messages
