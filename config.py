from pathlib import Path


# set configuration
config = {
    'seed': 25,
    'gpu_id': 0,
    'model_name': 'microsoft/deberta-v3-large',
    'optimizer': 'AdamW',
    'lr': 0.00008487135875395362,
    'epochs': 10,
    'epochs_stop': 3,
    'batch_size': 8,
    'hidden_size': 768,

    'max_seq_len': 1024,
    'num_labels': 1,
    'bert_output': 'pooler',

    'lora_r': 16,
    'lora_alpha': 8,
    'lora_dropout': 0.1,
    'lora_bias': 'all',

    'save_model': False,
    'model_path': Path('./sadness'),
    # 'pred_path': 'Prediction.csv',
    'train_path': 'train_set.csv',
    'dev_path': 'dev_set.csv',
    'test_path': 'test_set.csv',
    'x': 'text',
    'y': 'sadness'
    # 'special_tokens': ['[#CLIENT]', '[#COUNSELOR]', '[#DATE]', '[#PERSON]']
}

# after hyperparameter tuning, update config with the best performance for testing
'''
config.update(
    {
        "lr": 0.00007756700580384453,
        "optimizer": "Adam",
        "bert_output": "last_4",
        "lora_dropout": 0.24341246429453545,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_bias": "lora_only"
    }
)
'''
