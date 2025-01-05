# Fine-tune RoBERTa/DeBERTa models to predict emotion labels on text data
This project sets a pipeline to fine-tune pretrained large language models to automatically predict emotion labels on text data. Users can customize hyperparameters in the `config.py` and `tuner.py`. This work is a part of the dissertation project to examine the dynamics of suicide-related emotions and suicide risk from moment to moment. The chosen emotions include guilt, shame, loneliness, anger, sadness, depressed, and anxiety.

Models include Facebook RoBERTa and Microsoft DeBERTa (base and/or large, parameter sizes 100M+ to 300M+). Hyperparameter
tuning is performed using Microsoft NNI (Neural Network Intelligence) toolkit 2.5. Computational cost is optimized by using 
distributed training on multiple GPUs, automatic mixed precision (AMP) training, and low-rank adaptation (LoRA).

Special thanks to Maitrey Mehta and Mattia Medina-Grespan for their suggestions on codes.  

## Copy the repository
```bash
git clone https://github.com/XinY-Z/bert-text2emo
```

## Installation
```bash
pip install -r requirements.txt
```

## Data
The data used in this project came from following datasets:
- [ISEAR](https://paperswithcode.com/dataset/isear)
- [FIG-Loneliness](https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness)
- [Vent](https://paperswithcode.com/dataset/vent)
- [EmoInt](https://www.kaggle.com/code/ardacandra/wassa-2017-emotion-intensity)

## Usage
First, customize the `config.py` file to your own data. 
```python
config['train_path']: <your training data path>
config['dev_path']: <your development data path>
config['test_path']: <your test data path>
config['x'] = <your text column>
config['y'] = <your label column>
```

Second, customize the `tuner.py` file to your own hyperparameter tuning settings. For example, 
```python
# specify the search space of the models
experiment.config.search_space = {
    "model_name": {
        "_type": "choice",
        "_value": ["facebookAI/roberta-base", "facebookAI/roberta-large", "microsoft/deberta-base", "microsoft/deberta-large"]
    },
}
```

Third, run the `tuner.py` file to start hyperparameter tuning.
```bash
python3 tuner.py
# This will open a web interface to monitor the training process
```
Find the best hyperparameters from the NNI web interface and update the `config.py` file.

Fourth, run the `train.py` file to train the model with the best hyperparameters.
```bash
python3 train.py
```

Finally, check the model performance on the test set by running the `evaluate.py` file.
```bash
python3 evaluate.py
```
