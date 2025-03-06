import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultiTaskModel(nn.Module):
    def __init__(self, num_sentence_classes=2, num_ner_classes=5, model_name='bert-base-uncased'):
        super(MultiTaskModel, self).__init__()
        # Shared BERT backbone
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # Sentence classification head
        self.sentence_classifier = nn.Linear(hidden_size, num_sentence_classes)
        
        # NER classification head
        self.ner_classifier = nn.Linear(hidden_size, num_ner_classes)

    def forward(self, input_ids, attention_mask):
        # Get outputs from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Sentence classification: mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        sentence_logits = self.sentence_classifier(sentence_embeddings)  # Shape: (batch_size, num_sentence_classes)
        
        # NER classification: per-token predictions
        ner_logits = self.ner_classifier(last_hidden_state)  # Shape: (batch_size, seq_len, num_ner_classes)
        
        return sentence_logits, ner_logits

# Test the model
model = MultiTaskModel(num_sentence_classes=2, num_ner_classes=5)
sample_sentences = ["Let's get this party started!", "Humans will always win against AGI"]
encoding = model.tokenizer(sample_sentences, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
sentence_logits, ner_logits = model(input_ids, attention_mask)
print("Sentence Logits Shape:", sentence_logits.shape)
print("NER Logits Shape:", ner_logits.shape) 

# Shared Backbone: The BERT model remains shared to learn representations beneficial for both tasks
# Sentence Classification Head: Takes the pooled sentence embedding and outputs logits for 2 classes (e.g. +/-). I assumed a binary classification task for simplicity.
# NER Head: Operates on the full last_hidden_state to produce per-token logits for 5 NER classes 