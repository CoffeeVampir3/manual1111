import torch

@torch.no_grad()
def weigh_embeddings(tokenizer, prompt, text_embeddings, device, debug=False):
    #One major note is that there's an additional token at 1 and 77 which are system control tokens
    #So our index starts at index+1 because the first token is a control token.
    
    #Create an unweighted set of weights
    weights = torch.ones(1, 77, 1, device=device)

    tokens = tokenizer.tokenize(prompt)

    #add token weights using a simple moving sum of brackets
    bracket_weight = .1 #10% weight
    bracket_direction = 0
    debug_weights = [0]
    for index, token in enumerate(tokens):
        positive_direction = token.count('(') - token.count(')')
        negative_direction = token.count('[') - token.count(']')
        
        bracket_direction += positive_direction - negative_direction
        current_weight = 1.0 + bracket_direction * bracket_weight
        
        if positive_direction != 0 or negative_direction != 0: #weight brackets as weight of 0, in the future we should delete them from the prompt to save the token count.
            debug_weights.append(0)
            weights[0][index+1] = 0
            continue
        
        debug_weights.append(current_weight)
        weights[0][index+1] = current_weight

    #Re-mean the weights to restore the original mean.
    weighted_embeddings = text_embeddings * weights
    #weighted embedding can never be all 0 because the control tokens at worst will always be 1.
    weighted_embeddings *= text_embeddings.mean() / weighted_embeddings.mean()

    ### Debugging mode to see prompt weightings.
    if debug:
        debug_values = zip(tokens, debug_weights)
        for i,(x,y) in enumerate(debug_values):
            i = i + 1
            print((f"""
                    Index: {i:<2}
                    Token: {x:<18} 
                    Weight: {y}
                    Weight tensor: {weights[0][i].item()},
                    Embedding Mean: {weighted_embeddings[0][i].mean().item()}"""))

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

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]

    weighted_prompt_embeddings = weigh_embeddings(tokenizer, text, prompt_embeds, device)
    return pooled_prompt_embeds, weighted_prompt_embeddings

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
    
    pos_openclip_pool, pos_openclip = encode_line(positive_prompt, openclip_tokenizer, openclip_encoder, device)
    neg_openclip_pool, neg_openclip = encode_line(negative_prompt, openclip_tokenizer, openclip_encoder, device)

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
    pos_openclip_pool = pos_openclip_pool.repeat(repeats, 1, 1).view(repeats, -1)
    neg_openclip_pool = neg_openclip_pool.repeat(repeats, 1, 1).view(repeats, -1)
    
    return positives, negatives, pos_openclip_pool, neg_openclip_pool

def encode_from_pipe(pipe, pos_prompt, neg_prompt, pos_key, neg_key, repeats):
    clip_enc, clip_tok = pipe.text_encoder, pipe.tokenizer
    openclip_enc, openclip_tok = pipe.text_encoder_2, pipe.tokenizer_2
    return encode_all(pos_prompt, neg_prompt, pos_key, neg_key, clip_enc, clip_tok, openclip_enc, openclip_tok, repeats, pipe.device)
