#!/usr/bin/env python3

"""Do the World Value Survey with a model from the Hugging Face Hub."""

import argparse
from datetime import datetime
import gc
import json
import logging
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import transformers
import torch


def get_results_int(answer: str) -> Optional[int]:
    last_line = answer.strip().split("\n")[-1]
    result = None
    for token in reversed(last_line.split()):
        try:
            result = int(token)
            break
        except ValueError:
            pass
    return result


def valid_results_1_to_n(n: int) -> Callable[[str], Optional[int]]:
    def validator(answer: str) -> Optional[int]:
        result = get_results_int(answer)
        if result is not None and 1 <= result <= n:
            return result
        return None
    return validator


VALIDATORS = {
    "1or2": valid_results_1_to_n(2),
    "1to3": valid_results_1_to_n(3),
    "1to4": valid_results_1_to_n(4),
    "1to5": valid_results_1_to_n(5),
    "1to10": valid_results_1_to_n(10),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID from the Hugging Face Hub.",
    )
    parser.add_argument(
        "lng", default="en", help="Questinaire language.")
    parser.add_argument(
        "--max-history", type=int, default=80,
        help="Maximum number of messages to keep in history.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id, # type: ignore
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") # type: ignore
    ]

    all_messages = []
    messages = []
    results = {}

    def send_message(msg):
        prompt = pipeline.tokenizer.apply_chat_template( # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True
        )
        outputs = None
        while outputs is None:
            try:
                outputs = pipeline(
                    prompt,
                    max_new_tokens=2000,
                    eos_token_id=terminators,
                    pad_token_id=pipeline.tokenizer.pad_token_id, # type: ignore
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                )
            except RuntimeError as e:
                # If e is OOM, remove a message from the beginning of the conversation
                if "CUDA out of memory" in str(e):
                    logging.warning("CUDA out of memory, removing a message from the beginning of the conversation.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if messages:
                        messages.pop(0)
                    else:
                        logging.error("No more messages to remove, cannot recover from OOM.")
                        exit(1)
        generated_text = outputs[0]["generated_text"][len(prompt):] # type: ignore
        return generated_text

    with open(f"questions.{args.lng}.txt") as f, open("validators.txt") as v:
        questions = f.readlines()
        validators = v.readlines()

    for question, validator in zip(questions, validators):
        q_id, question = question.strip().split(";", maxsplit=1)
        logging.info(f"Processing {q_id}: {len(results) + 1} / {len(questions)}")
        validator = validator.strip()
        messages.append({"role": "user", "content": question})
        all_messages.append({"role": "user", "content": question})
        start_time = datetime.now()
        attempts = 0
        while results.get(q_id) is None:
            attempts += 1
            answer = send_message(question)
            results[q_id] = VALIDATORS[validator](answer) # type: ignore
            if attempts > 50:
                logging.warning("Too many attempts, skipping to the next question.")
                answer = ""
                break
        messages.append({"role": "assistant", "content": answer})
        all_messages.append({"role": "assistant", "content": answer})
        duration = datetime.now() - start_time

        logging.info("The last prompt took %s with %d attempts." % (duration, attempts))
        logging.info("History has %d messages." % len(messages))
        logging.info("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        logging.info("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        logging.info("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        if len(messages) > args.max_history:
            messages = messages[-args.max_history:]

    # Print results as a JSON object
    print(json.dumps({"messages": all_messages, "results": results}, indent=2))


if __name__ == "__main__":
    main()
