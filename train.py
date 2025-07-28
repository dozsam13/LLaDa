import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import download_text, create_dataset_from_text
from tokenizer import LLADACharacterTokenizer
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from model import MaskPredictor
import argparse
import json
from statistics import mean


seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def forward_process(sequences, t, mask_token_id, device):
    '''
    Mask every token with t probability.
    '''
    mask = torch.rand(sequences.size()).to(device)
    mask = mask < t
    masked_sequences = sequences.masked_fill(mask, mask_token_id)

    return masked_sequences.int(), mask.bool()


def compute_llada_loss(model, sequences, t, mask_token_id, device, compute_accuracy=False):
    '''
    Compute loss and accuracy on the given batch of sequences masked with t probability.
    '''
    masked_sequences, mask = forward_process(sequences, t, mask_token_id, device)
    prediction = model(masked_sequences)
    prediction = prediction.permute(0, 2, 1)

    masked_target = sequences * mask
    sequences[~mask] = -1000

    prediction = prediction.float()
    weighted_loss = nn.CrossEntropyLoss(ignore_index=-1000)(prediction.float(), sequences.long()) * 1/(t+0.05)
    if not compute_accuracy:
        return weighted_loss, None
    else:
        pred_tokens = torch.argmax(prediction, dim=-2)

        correct = torch.sum(pred_tokens[mask] == sequences[mask])
        all = torch.sum(mask)
        accuracy = correct/all

        return weighted_loss, accuracy


def get_lr(it, config):
    '''
    Calculate learning rate for the given epoch.
    '''
    if it < config['warmup_iters']:
        return config['learning_rate'] * (it + 1) / (config['warmup_iters'] + 1)
    if it > config['lr_decay_iters']:
        return config['min_lr']
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])



@torch.no_grad()
def estimate_loss_accuracy(model, dataset, batch_size, device, mask_token_id, iters=1):
    '''
    Estimate loss and accuracy on the given dataset with random masking probabilities
    '''
    losses = []
    model.eval()
    accuracies = []
    for _ in range(iters):
        for i in range(0, len(dataset)-batch_size, batch_size):
            t = random.uniform(0, 1)
            loss, accuracy = compute_llada_loss(model, dataset[i:i+batch_size], t, mask_token_id, device, compute_accuracy=True)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy.cpu().item())

    model.train()

    return mean(losses), mean(accuracies)

def pretrain_llada(model, train_dataset, val_dataset, mask_token_id, device, train_config, use_wandb):
    '''
    Train MaskPredictor on the given dataset.
    '''
    if use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project='llada_pretrain',
            config=train_config
        )
    os.makedirs("checkpoints", exist_ok=True)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.95, 0.99),lr=1e-3)
    for epoch in tqdm(range(train_config['num_epochs'])):
        train_loss = []

        # Set lr
        lr = get_lr(epoch, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        for batch_data in DataLoader(train_dataset, train_config['batch_size'], shuffle=True):
            t = random.uniform(0, 1)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_llada_loss(model, batch_data, t, mask_token_id, device)
            loss.backward()
            optimizer.step()

        if epoch % train_config['eval_interval'] == 0 and epoch != 0 or epoch == train_config['num_epochs']-1:
            train_loss, train_accuracy = estimate_loss_accuracy(model, train_dataset, train_config['batch_size'], device, mask_token_id, iters=15)
            val_loss, val_accuracy = estimate_loss_accuracy(model, val_dataset, train_config['batch_size'], device, mask_token_id, iters=15)
            if use_wandb:
                wandb.log({'Loss/train': train_loss, 'Loss/val':val_loss, 
                          'Accuracy/train': train_accuracy, 'Accuracy/val': val_accuracy})

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)

            print(f"Epoch {epoch} Train loss: {train_loss} Val loss: {val_loss} train_acc:{train_accuracy} val_acc:{val_accuracy}")

            with open("history.json", "w") as f:
                json.dump(history, f)
            if val_accuracy > best_val_acc or epoch == train_config['num_epochs']-1:
                best_val_acc = val_accuracy
                save_checkpoint(model, optimizer, epoch, best_val_acc, use_wandb)
                

    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.savefig('loss.png')
    plt.show()
    plt.clf()
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Accuracy')
    plt.savefig('accuracy.png')
    plt.show()
    plt.clf()

    return model

def save_checkpoint(model, optimizer, epoch, best_val_acc, use_wandb):
    checkpoint = {
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'iter_num': epoch,
          'best_val_acc': best_val_acc
    }
    out_dir = f"checkpoints/{str(epoch)}.pt"
    print(f" * Saving checkpoint to {out_dir}")
    torch.save(checkpoint, out_dir)
    if use_wandb:
        artifact = wandb.Artifact(name=f"model-ckpt-{epoch}", type="model")
        artifact.add_file(out_dir)
        wandb.log_artifact(artifact)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_argparser():
    parser = argparse.ArgumentParser(description="Script to download dataset and train LLaDa model")

    parser.add_argument("--config", type=str, help="Path to train config file")
    parser.add_argument("--wandb", action='store_true', help="Use weights&biases for experiment tracking")

    args = parser.parse_args()

    return args


def main():
    args = create_argparser()
    train_config = load_config(args.config)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_text = download_text(train_config["text_url"])
    tokenizer = LLADACharacterTokenizer(raw_text)
    train_dataset, val_dataset = create_dataset_from_text(raw_text, tokenizer, 
                                                          train_config['context_size'],
                                                          DEVICE)

    mask_predictor = MaskPredictor(train_config['context_size'], 
                                   train_config['embd_size'], 
                                   tokenizer.vocab_size)
    mask_predictor.to(DEVICE)

    model_parameter_count = sum(p.numel() for p in mask_predictor.parameters() if p.requires_grad)
    print(f'Model size: {model_parameter_count} parameters')

    mask_predictor = pretrain_llada(mask_predictor, train_dataset, val_dataset, 
                                    tokenizer.mask_token_id, DEVICE, train_config, args.wandb)

if __name__=="__main__":
    main()


