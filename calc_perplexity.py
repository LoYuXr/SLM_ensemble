import torch

def compute_perplexity(model, tokenizer, text, device="cuda"):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        # Get model outputs (logits)
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Calculate cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Calculate perplexity
        ppl = torch.exp(loss.mean())
    return ppl.item()
