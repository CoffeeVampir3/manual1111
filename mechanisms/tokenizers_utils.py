import torch

@torch.no_grad()
def encode_line(text, tokenizer, text_encoder, device):
    text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(text, padding="longest", return_tensors="pt").input_ids

    prompt_embeds = text_encoder(
        text_input_ids.to(device),
        output_hidden_states=True,
    )
    return prompt_embeds[0], prompt_embeds.hidden_states[-2]

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