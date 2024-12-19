## Adapted From https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_gsm8k.py
import json
import re
from pathlib import Path
import argparse
import requests
import math
import numpy as np
import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
# from loguru import logger
from utils import *

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default="")
    args = parser.parse_args()

    acc_res = []

    for x in tqdm.tqdm(os.listdir(args.folder)):
        path = os.path.join(args.folder, x)
        data = load_json(path)
        completion = data['completion']
        answer = data['answer']
        acc = is_correct(completion, answer)
        acc_res.append(acc)

    print("Acc: ", np.mean(acc_res))