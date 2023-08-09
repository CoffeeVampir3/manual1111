import torch

def weigh_embeddings(tokenizer, prompt, text_embeddings, device):
    #Create an unweighted set of weights
    weights = torch.full(text_embeddings.shape, 1.0, device=device)

    tokens = tokenizer.tokenize(prompt)

    #add token weights using a simple moving sum of brackets
    current_weight = 1.0
    bracket_weight = 5.0
    any_weights = False
    for index, token in enumerate(tokens):
        positive_direction = token.count('(') + token.count(']')
        negative_direction = token.count('[') + token.count(')')
        
        bracket_count = positive_direction - negative_direction
        current_weight += bracket_count * bracket_weight
        
        if bracket_count == 0:
            weights[..., index+1] = current_weight
            any_weights = True
            print(token, current_weight)

    if not any_weights:
        return text_embeddings
    
    #Re-mean the weights to restore the original mean.
    weighted_embeddings = text_embeddings * weights
    normalization_factor = text_embeddings.mean() / weighted_embeddings.mean()
    weighted_embeddings *= normalization_factor

    return weighted_embeddings

@torch.no_grad()
def encode_line(text, tokenizer, text_encoder, device):
    text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(text, padding="longest", return_tensors="pt").input_ids

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        output_hidden_states=True,
    )
    
    weighted = weigh_embeddings(tokenizer, text, prompt_embeds[0], device)
    return weighted, prompt_embeds.hidden_states[-2]

@torch.no_grad()
def encode_all(
    positive_prompt,
    negative_prompt,
    positive_keywords,
    negative_keywords,
    clip_encoder,
    clip_tokenizer,
    openclip_encoder,
    openclip_tokenizer,
    repeats,
    device
):
    _, pos_clip = encode_line(positive_keywords, clip_tokenizer, clip_encoder, device)
    _, neg_clip = encode_line(negative_keywords, clip_tokenizer, clip_encoder, device)
    
    pos_pool, pos_openclip = encode_line(positive_prompt, openclip_tokenizer, openclip_encoder, device)
    neg_pool, neg_openclip = encode_line(negative_prompt, openclip_tokenizer, openclip_encoder, device)

    pos_encodings = [
        pos_clip,
        pos_openclip
    ]

    neg_encodings = [
        neg_clip,
        neg_openclip
    ]

    positives = torch.concat(pos_encodings, dim=-1).repeat(repeats, 1, 1)
    negatives = torch.concat(neg_encodings, dim=-1).repeat(repeats, 1, 1)
    pos_pool = pos_pool.repeat(repeats, 1, 1).view(repeats, -1)
    neg_pool = neg_pool.repeat(repeats, 1, 1).view(repeats, -1)
    return positives, negatives, pos_pool, neg_pool

def encode_from_pipe(pipe, pos_prompt, neg_prompt, pos_key, neg_key, repeats):
    clip_enc, clip_tok = pipe.text_encoder, pipe.tokenizer
    openclip_enc, openclip_tok = pipe.text_encoder_2, pipe.tokenizer_2
    return encode_all(pos_prompt, neg_prompt, pos_key, neg_key, clip_enc, clip_tok, openclip_enc, openclip_tok, repeats, pipe.device)