from transformers import RobertaModel
from torch import nn, optim


class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, PRE_TRAINED_MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(pooled_output)
        return self.out(output)
