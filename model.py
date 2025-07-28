from transformers import BertModel, BertConfig
import torch.nn as nn


class MaskPredictor(nn.Module):
    def __init__(self, context_size, embedding_size, vocab_size):
        super().__init__()
        bert_config = BertConfig(vocab_size=vocab_size, hidden_size=context_size,
                                 num_hidden_layers = 8, num_attention_heads = 8,
                                 intermediate_size = embedding_size)
        
        self.bert_model = BertModel(bert_config)
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx):
        x = self.bert_model(idx).last_hidden_state 
        x = self.ln_f(x)
        
        logits = self.lm_head(x)

        return logits