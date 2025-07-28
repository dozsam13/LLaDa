import torch.nn as nn
import math
import torch
import argparse
from model import MaskPredictor
import json
from dataset import download_text
from tokenizer import LLADACharacterTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def logits_to_token(logits, sample_top_k=3, do_sampling=False):
    '''
    Get actual token from predicted logits.
    '''
    prob = nn.functional.softmax(logits, dim=-1)
    if do_sampling:
        values, indices = torch.topk(logits, sample_top_k, largest=True, dim=-1)

        dist = nn.functional.softmax(values, dim=-1)
        sampled_indices = torch.multinomial(dist[0], num_samples=1)

        sampled_indices = sampled_indices.unsqueeze(0)
        predicted_tokens = torch.gather(indices, dim=2, index=sampled_indices)
        c = torch.gather(dist, dim=2, index=sampled_indices)
        return predicted_tokens.squeeze(-1).int(), c.squeeze(-1)
    else:
        values, indices = torch.max(prob, dim=-1)
        return indices.int(), values

def get_new_mask(mask, fraction_to_remask, probabilities, mask_n, low_conf_remask=False):
    '''
    Apply remasking strategy.
    '''
    if low_conf_remask:
        probabilities[~mask] = 1
        _, indices = torch.topk(probabilities, mask_n, largest=False)
        mask[:, :] = False
        mask[:, indices[0]] = True
        return mask
    else:
        rnd = torch.rand(mask.size()).to(DEVICE)
        rnd = rnd < fraction_to_remask
        return mask * rnd


@torch.no_grad()
def reverse_process_generate(model, prompt, max_length, num_steps, tokenizer, do_sampling, low_conf_remask, sample_top_k=3):
    '''
    Reverse process to generate text for prompt.
    '''
    model.eval()
    sequence = torch.full(size=(1, max_length), fill_value=tokenizer.mask_token_id, device=DEVICE, dtype=torch.int)
    prompt = tokenizer.encode(prompt)
    sequence[:, 0:len(prompt)] = torch.Tensor(prompt)
    mask = sequence == tokenizer.mask_token_id
    for step in range(num_steps):
        t = (num_steps - step) / num_steps
        s = (num_steps - step - 1) / num_steps

        sequence[mask] = tokenizer.mask_token_id
        logits = model(sequence)
        predicted_tokens, C = logits_to_token(logits, sample_top_k, do_sampling = do_sampling)
        sequence[mask] = predicted_tokens[mask]

        fraction_to_remask = s / t

        L = max_length-len(prompt)
        nun = math.floor(L*(1-s))
        mask = get_new_mask(mask, fraction_to_remask, C, L-nun, low_conf_remask)

    return tokenizer.decode(sequence.tolist()[0])


def create_argparser():
    parser = argparse.ArgumentParser(description="Script to do reverse process with a trained LLaDa model")

    parser.add_argument("--config", type=str, required=True, help="Path to train config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for text generation")

    parser.add_argument("--num_steps", type=int, default=50, help="Number of reverse process steps")
    parser.add_argument("--do_sampling", type=bool, default=False, help="Do distribution sampling")
    parser.add_argument("--low_conf_remask", type=bool, default=True, help="Use low confidence remasking strategy")


    args = parser.parse_args()

    return args


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main():
    args = create_argparser()
    config = load_config(args.config)

    raw_text = download_text(config["text_url"])
    tokenizer = LLADACharacterTokenizer(raw_text)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device(DEVICE), weights_only=False)
    mask_predictor = MaskPredictor(config['context_size'], 
                                   config['embd_size'], 
                                   tokenizer.vocab_size)
    mask_predictor.to(DEVICE)
    
    mask_predictor.load_state_dict(checkpoint['model'], strict=False)
    mask_predictor.eval()

    result = reverse_process_generate(mask_predictor, args.prompt, config['context_size'], args.num_steps,
                                  tokenizer, do_sampling=args.do_sampling, low_conf_remask=args.low_conf_remask)

    print(f"Generated text:\n----\n{result}\n----")


if __name__=="__main__":
    main()