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
from calc_acc import extract_answer
from calc_perplexity import compute_perplexity

## Maybe Distributed to run faster
world_size = int(os.getenv("WORLD_SIZE", "-1"))
rank = int(os.getenv("RANK", "-1"))
device = "cuda:1"

if world_size > 0 and rank != -1:
    device = f"cuda:{rank}"
    print(f"Rank: {rank}, Device: {device}")

def doc_to_text(doc, use_fewshot):
    if use_fewshot:
        context = (
            "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\n"
            "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\n"
            "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\n"
            "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n\n"
            "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\n"
            "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n\n"
            "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\n"
            "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n\n"
            f"Question: {doc['question']}\nLet's think step by step"
        )
    else:
        context = doc["question"]
    return context


'''
对于以下代码，我有需求：
将其改成voting方法，即对于每一个测试数据，生成num_sample个数的回答，然后batch_decode出一系列的回答，对于每个回答处理得到最终的数值答案，并统计num_sample中个数最多的答案作为最终的答案然后进行统计，计算acc.
'''

def batch_chat(model, tok, ques, history=[], num_sample=5):
    iids = tok.apply_chat_template(
        history + [{'role': 'user', 'content': ques}], 
        add_generation_prompt=1,
    )
    oids = model.generate(
        inputs=torch.tensor([iids]*num_sample).to(model.device),
        **(model.generation_config.to_dict()),
    )
    oids = oids[:, len(iids):].tolist()
    ans = tok.batch_decode(oids)
    # ans is a list of strings
    return ans


def chat(model, tok, ques, history=[]):
    iids = tok.apply_chat_template(
		history + [{'role': 'user', 'content': ques}], 
		add_generation_prompt=1,
	)
    oids = model.generate(
		inputs=torch.tensor([iids]).to(model.device),
		**(model.generation_config.to_dict()),
	)
    oids = oids[0][len(iids):].tolist()
    if oids[-1] == tok.eos_token_id:
        oids = oids[:-1]
    ans = tok.decode(oids)
    return ans

def voting(model, tokenizer, question, num_samples=5, save_name=None, perplexity=False, device=None):
    
    answer_counts = {}
    answer_num = {}
    iters = 0
    while True:
        responses = batch_chat(model, tokenizer, question, num_sample=num_samples)
        num_answers = [extract_answer(response) for response in responses]
        if perplexity:
            ppl = [compute_perplexity(model, tokenizer, response,device=device) for response in responses]
           
            # find the least ppl id and use it as the answer
            min_ppl_id = ppl.index(min(ppl))
            response = responses[min_ppl_id]
            print(f"Perplexity: {ppl}")
            print(f"Save_name: {save_name}, Min ppl id: {min_ppl_id}\n")
            return response

        for number, response in zip(num_answers, responses):
            if number is None:
                continue
            if number not in answer_counts:
                answer_counts[number] = []
                answer_num[number] = 0
                
            answer_counts[number].append(response)
            answer_num[number] += 1
        
        max_key = max(answer_num, key=answer_num.get)
        min_key = min(answer_num, key=answer_num.get)
        if iters > 5 or (len(answer_num.keys()) == 1 and max_key == min_key):
            break
        elif (answer_num[max_key]>1 and answer_num[max_key] > answer_num[min_key]):
            break
        else:
            iters +=1

    print(f"Save_name: {save_name}")
    print(num_answers)
    print(answer_num)
    print(f"Save_name: {save_name}, Max key: {max_key}\n")

    response = answer_counts[max_key][0]
    # print(question)
    # print("-------------")
    # print(response)
    # print("=============")
    return response


def generate_sample(model, tokenizer, question):
    response = chat(model, tokenizer, question)
    print(question)
    print("-------------")
    print(response)
    print("=============")
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default="/home/luoyx/WORK/MLHW/gsm8k")
    parser.add_argument("--model_path",default="/home/luoyx/WORK/MLHW/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--save_path", type=str, default="/home/luoyx/WORK/MLHW/SLM_ensemble/gsm8k_perplex")
    parser.add_argument("--use_fewshot", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--perplexity", default=False)
    args = parser.parse_args()

    save_path = os.path.join(args.save_path, "zeroshot") if args.use_fewshot else os.path.join(args.save_path, "fewshot")
    os.makedirs(save_path,exist_ok=True)
    dataset = load_dataset(args.data_path, "main")

    ## Method Part
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, bf16=True, use_flash_attn=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).eval().to(device)
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    model.generation_config.do_sample = True  # use greedy decoding
    model.generation_config.repetition_penalty = 1  # disable repetition penalty
    model.generation_config.temperature = 0.7
    model.generation_config.max_new_tokens = 512

    ## Test Part
    test = dataset["test"]
    for idx, doc in tqdm.tqdm(enumerate(test)):
        if world_size != -1:
            if idx % world_size != rank:
                continue
        save_name = os.path.join(save_path, f"{idx}.json")
        if os.path.exists(save_name):
            #logger.info(f'Exists {idx} and Skip!')
            print(f'Exists {idx} and Skip!')
            continue
        context = doc_to_text(doc, args.use_fewshot)
        if args.num_samples > 1:
            completion = voting(model, tokenizer, context, num_samples=args.num_samples, save_name=save_name, perplexity=args.perplexity, device=device)
        else:
            completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        saved = {
            'question': doc['question'],
            'completion': completion,
            'answer': doc['answer']
        }
        save_json(saved, save_name)
