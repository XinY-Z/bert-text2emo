import torch
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# set model
class BERT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(self.config['model_name'])
        self.peft_config = LoraConfig(r=self.config['lora_r'],
                                      lora_alpha=self.config['lora_alpha'],
                                      lora_dropout=self.config['lora_dropout'],
                                      bias=self.config['lora_bias'])
        self.peft_bert = get_peft_model(self.bert, self.peft_config)
        self.hidden_size = self.config['hidden_size']
        self.mlp = torch.nn.Linear(self.hidden_size, self.config['num_labels'])
        self.pooler = Pooler(self.hidden_size)
        '''if config['dropout_prob'] > 0:
            self.dropout = torch.nn.Dropout(config['dropout_prob'])'''

    def forward(self, **kwargs):
        """
        :param input_ids: Tensor, shape ``[batch_size, seq_len]``
        :param attention_mask: Tensor, shape ``[batch_size, seq_len]``
        :return: Tensor, shape ``[batch_size, num_labels]``
        """

        # get output from BERT
        if self.config['bert_output'] == 'pooler':
            outputs = self.peft_bert(output_hidden_states=True, **kwargs)
            outputs.pooler_output = self.pooler(outputs.hidden_states)
            embeddings = outputs.pooler_output
        elif self.config['bert_output'] == 'pooled_mean':
            outputs = self.peft_bert(output_hidden_states=True, **kwargs)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        elif self.config['bert_output'] == 'last_4':
            outputs = self.peft_bert(output_hidden_states=True, **kwargs)
            embeddings = torch.stack([outputs.hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]], dim=1).mean(dim=1)
        else:
            raise ValueError('bert_output must be one of "pooler", "pooled_mean", "last_4"')

        # add dropout layer
        '''if self.config['dropout_prob'] > 0:
            embeddings = self.dropout(embeddings)'''

        # get MLP output
        logits = self.mlp(embeddings)

        return logits


# set DeBERTa pooler output layer
class Pooler(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[-1][:, 0, :]
        pooler_output = self.mlp(first_token_tensor)
        pooler_output = self.activation(pooler_output)

        return pooler_output
