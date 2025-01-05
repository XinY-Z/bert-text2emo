import torch
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import nni
import gc
from preprocessor import read_data
from model import BERT
from config import config


# set environment
torch.cuda.set_device(config['gpu_id'])
device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config['seed'])


# set training
def train(train_dataloader, dev_dataloader, config):
    # use scaler to prevent gradient underflow when using automatic mixed precision (AMP)
    scaler = torch.amp.GradScaler('cuda')

    # check if num_labels is consistent
    assert config['num_labels'] == train_dataloader.dataset.y.shape[1] == dev_dataloader.dataset.y.shape[1]

    # get tuner parameters
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    print(f'tuner_params: {tuner_params}')

    # initiate tokenizer and training model
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(BERT(config)).to(device)
    else:
        model = BERT(config).to(device)

    # initiate weighted loss function
    total = len(train_dataloader.dataset)
    weights = [0 for _ in range(config['num_labels'])]
    for i in range(config['num_labels']):
        count = np.sum([x[i] for x in train_dataloader.dataset.y])
        weights[i] = (total - count) / count
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights).to(device))

    # initiate optimizer
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    else:
        raise ValueError(f'Invalid optimizer name: {config["optimizer"]}')

    # initiate trainer
    dev_metrics = {}
    best_metrics = {'f1_macro': 0.}
    best_epoch = 0
    best_state_dict = {}
    best_dev_metrics = {}
    no_improve = 0

    # training loop
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = .0
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            embeddings = tokenizer(list(x), padding=True, truncation=True, max_length=config['max_seq_len'],
                                   return_tensors='pt')
            # use automatic mixed precision (AMP) to reduce memory usage
            with torch.amp.autocast('cuda'):
                logits = model(**embeddings.to(device))
                loss = loss_fn(logits, y.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            '''logits = model(**embeddings.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optimizer.step()'''
            total_loss += loss.item()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        ave_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch} loss: {ave_loss}')

        dev_metrics.update(evaluate(dev_dataloader, tokenizer, model))
        print(
            f"""Epoch {epoch} dev 
             f1_macro: {dev_metrics["f1_macro"]}
             recall: {dev_metrics["recall"]}
             precision: {dev_metrics["precision"]}
             pr_auc: {dev_metrics["pr_auc"]}
             roc_auc: {dev_metrics["roc_auc"]}"""
        )

        nni.report_intermediate_result(dev_metrics['f1_macro'])

        if dev_metrics['f1_macro'] > best_metrics['f1_macro']:
            epsilon = dev_metrics['f1_macro'] - best_metrics['f1_macro']
            best_metrics.update(dev_metrics)
            best_epoch = epoch
            best_state_dict = {
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }
            best_dev_metrics = json.dumps(dev_metrics, indent=4)
            if epsilon >= .01:
                no_improve = 0
            else:
                no_improve += 1
            # no_improve = 0

        elif best_metrics['f1_macro'] != 0:
            no_improve += 1

        if no_improve == config['epochs_stop']:
            print(f'Early stopping at epoch {epoch}.')
            break

    nni.report_final_result(dev_metrics['f1_macro'])

    # save best model
    if config['save_model']:
        model_folder = Path(config['model_path'], f'seed{config["seed"]}')
        Path.mkdir(model_folder, exist_ok=True, parents=True)
        torch.save(best_state_dict, Path(model_folder, f'epoch{best_epoch}.pt'))
        with open(Path(model_folder, f'epoch{best_epoch}_metrics.json'), 'w+') as f:
            f.write(best_dev_metrics)
        print(f'Epoch {best_epoch} model saved.')

    return None


# set evaluator
def evaluate(dataloader, tokenizer, model):
    metrics = {}

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            embeddings = tokenizer(list(x), padding=True, return_tensors='pt')
            with torch.amp.autocast('cuda'):
                logits = model(**embeddings.to(device))
            y_true.extend(y.tolist())
            y_pred.extend(logits.sigmoid().round().tolist())

    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['pr_auc'] = average_precision_score(y_true, y_pred, average='macro')
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    return metrics


if __name__ == '__main__':
    # reset environment
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # read data
    train_dataloader = read_data(config, 'train')
    dev_dataloader = read_data(config, 'dev')

    # train model
    train(train_dataloader, dev_dataloader, config)

    # clean up
    del train_dataloader, dev_dataloader
    torch.cuda.empty_cache()
    gc.collect()
