import json
from pathlib import Path
import time
import torch
from transformers import TextStreamer
from typing import Literal
from tools import print_message


def get_answer(
    question: str,
    answer,
    model,
    tokenizer,
    verbose: bool = True,
) -> str:
    model.eval()

    print_message(f"Q: {question}")
    print_message(f"GT Answer: {answer}")

    prompt = ""
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].size(1)

    t_start = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Extract the new tokens generated (excluding the prompt)
    output_tokens = tokens[:, prompt_length:]

    content = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    t = time.time() - t_start
    n_tokens = len(tokenizer.encode(content))
    stats = ""
    if verbose:
        stats = f"Time: {t:.2f}s, Tokens: {n_tokens}, Speed: {n_tokens / t:.2f} tokens/s"
    print_message("Network answer: " + content + "\n" + stats)
    return content


class Evaluator:
    def __init__(self, fname: str):
        with open(fname) as f:
            self.json_data = json.load(f)

    def evaluate(self, item: dict, answer: str) -> bool:
        response = answer.lower()
        if "answer" in item:
            return item["answer"].lower() in response

        if "any" in item:
            return any(
                phrase.lower() in response
                for phrase in item["any"]
            )

        if "all" in item:
            return all(
                phrase.lower() in response
                for phrase in item["all"]
            )

    def evaluate_all(self, get_answer_fn: callable, verbose: bool = False) -> float:
        n_correct = 0
        for item in self.json_data:
            dict_keys = list(item.keys())
            answer_n = dict_keys[1] if dict_keys[0] == "question" else dict_keys[0]
            answer = get_answer_fn(item["question"], item[answer_n])
            score = self.evaluate(item, answer)
            n_correct += score

            if verbose:
                print_message(f"Score: {score}")

        accuracy = n_correct / len(self.json_data)
        return accuracy

