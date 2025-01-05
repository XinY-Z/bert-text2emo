import torch
from transformers import AutoTokenizer
import gc
from pathlib import Path
import json

from train import evaluate
from preprocessor import read_data
from model import BERT
from config import config


# set environment
torch.cuda.set_device(config['gpu_id'])
device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config['seed'])

# change model name to saved model
config['saved_model_name'] = Path('./AngerModels/seed25/epoch3.pt')

# reset environment
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

# load data and saved model
test_dataloader = read_data(config, 'test')
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(BERT(config)).to(device)
else:
    model = BERT(config).to(device)
model.load_state_dict(torch.load(config['saved_model_name'])['model_state_dict'])

# evaluate the model
metrics = evaluate(test_dataloader, tokenizer, model)
json_metric = json.dumps(metrics, indent=4)
model_folder = Path(config['model_path'])
with open(Path(model_folder, f'test_metrics.json'), 'w+') as f:
    f.write(json_metric)
print(
    f"""test
    f1_macro: {metrics["f1_macro"]}
    recall: {metrics["recall"]}
    precision: {metrics["precision"]}
    pr_auc: {metrics["pr_auc"]}
    roc_auc: {metrics["roc_auc"]}
    """
)

# clean up
del test_dataloader, tokenizer, model
torch.cuda.empty_cache()
gc.collect()


if __name__ == '__main__':
    pass
