# SLM_ensemble
homework for ML2024

Model: `Qwen/Qwen2.5-0.5B-Instruct`

Download:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen2.5-0.5B-Instruct
```

Baseline:
- Voting
- Uncertainty
- Decomposition
- Critic

Dataset: `GSM8K`, 1000+ cases, maybe too much.

Download:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download openai/gsm8k --local-dir gsm8k
```

Run model
```
python run_gsm8k.py # default
python vote_gsm8k.py # voting
python vote_gsm8k.py --perplexity True #least perplexity
```

Test Acc
```
python calc_acc.py -f "/vepfs/zekai/SLM_ensemble/gsm8k/fewshot"
```
Naive Fewshot: 46.63
Voting: 52.84
Perplexity: 42.21
