#!/usr/bin/env python3
"""Do the World Value Survey with a model from the Hugging Face Hub."""
import argparse
from datetime import datetime
import gc
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_results_int(answer: str) -> Optional[int]:
    last_line = answer.strip().split("\n")[-1]
    result = None
    for token in reversed(last_line.split()):
        # Strip punctuation
        token = token.strip(".,;:!?()[]{}<>\"'`")
        try:
            result = int(round(float(token)))
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID from the Hugging Face Hub.",
    )
    parser.add_argument(
        "lng", default="en",
        help="Questionnaire language.")
    parser.add_argument(
        "prompt_type",
        choices=["cot", "score"],
        help="The type of the prompt.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.")
    parser.add_argument(
        "--greedy", action="store_true", default=False,
        help="Use greedy decoding instead of sampling.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Set up generation config
    generation_config = model.generation_config
    generation_config.do_sample = not args.greedy
    generation_config.temperature = 1.0 if args.greedy else 0.7
    generation_config.top_p = None if args.greedy else 0.9
    generation_config.max_new_tokens = 2000
    generation_config.pad_token_id = tokenizer.pad_token_id

    # Set up EOS tokens
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)
    generation_config.eos_token_id = terminators

    results = {}
    past_key_values = None
    conversation = []
    current_context = []

    def tokenize_context() -> torch.Tensor:
        """Tokenize the current context."""
        context = tokenizer.apply_chat_template(
            current_context,
            tokenize=True,
            truncation=True,
            truncation_side="left",
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return context

    def run_model() -> tuple[str, Optional[torch.Tensor]]:
        nonlocal past_key_values
        inputs = tokenize_context()
        if inputs.size(1) > 7000:
            while inputs.size(1) > 4000:
                # If the context is too long, remove the oldest message
                logging.info("The context is too long, removing the oldest message.")
                current_context.pop(0)
                inputs = tokenize_context()
                past_key_values = None
        inputs = inputs.to(model.device)

        try:
            outputs = model.generate(
                inputs,
                past_key_values=past_key_values,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
            )
        except RuntimeError:
            outputs = model.generate(
                inputs,
                past_key_values=None,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
            )

        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(
            outputs.sequences[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text, outputs.past_key_values

    with open(f"prompts/{args.prompt_type}.{args.lng}.txt", "r") as f, open("validators.txt", "r") as v:
        questions = f.readlines()
        validators = v.readlines()


    for question, validator in zip(questions, validators):
        q_id, question = question.strip().split(";", maxsplit=1)
        logging.info(f"Processing {q_id}: {len(results) + 1} / {len(questions)}")
        logging.info("Question: %s" % question)
        validator = validator.strip()

        quest_dict = {"role": "user", "content": question}
        conversation.append(quest_dict)
        current_context.append(quest_dict)

        start_time = datetime.now()
        attempts = 0
        answer = ""

        while results.get(q_id) is None:
            attempts += 1
            answer, maybe_past_key_values = run_model()
            results[q_id] = VALIDATORS[validator](answer)
            if results[q_id] is not None:
                past_key_values = maybe_past_key_values
                answer_dict = {"role": "assistant", "content": answer}
                conversation.append(answer_dict)
                current_context.append(answer_dict)
                break

            # If the answer is not valid, try again
            if attempts > (2 if args.greedy else 20):
                logging.warning("Too many attempts, skipping to the next question.")
                # Pretend the question was never asked for the context
                current_context.pop()
                # For the log, admit we do not know the answer
                conversation.append(None)
                break

        duration = datetime.now() - start_time
        logging.info("Answer: %s" % answer)
        logging.info("The last prompt took %s with %d attempts." % (duration, attempts))
        if past_key_values is not None:
            logging.info("The current context is %d tokens." % past_key_values[0][0].shape[2])
        else:
            logging.info("There is no past key values.")
        logging.info("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))
        logging.info("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0)/1024/1024/1024))
        logging.info("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    # Print results as a JSON object
    print(json.dumps({"messages": conversation, "results": results}, indent=2))

if __name__ == "__main__":
    main()
