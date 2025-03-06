import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # Get token embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Mean pooling: compute the mean across the sequence length dimension
        # Use attention_mask to exclude padding tokens from the mean
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
        sentence_embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_size)
        
        return sentence_embeddings

    def encode(self, sentences):
        # Tokenize input sentences
        encoding = self.tokenizer(sentences, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings = self.forward(input_ids, attention_mask)
        return embeddings

# Test the implementation
model = SentenceTransformer()
sample_sentences = [
    "Hello World!",
    "I'm AGI! How can I help you?"
]
embeddings = model.encode(sample_sentences)
print("Embeddings:", embeddings)
print("Sentence Embeddings Shape:", embeddings.shape)  # Expected: (2, 768)
for sentence, emb in zip(sample_sentences, embeddings):
    print(f"Sentence: '{sentence}' | Embedding (first 5 values): {emb[:5]}")

# Explanation:

# Transformer Backbone: I chose BERT (bert-base-uncased) because it’s a widely-used, pre-trained transformer model 
# that captures rich contextual information, making it suitable as a foundation for sentence embeddings. 
# Using a pre-trained model saves time compared to training a transformer from scratch and leverages BERT’s general language understanding.

#Pooling Strategy: To convert token-level embeddings from BERT into a single sentence embedding, 
# I used mean pooling over the token embeddings which aggregates information from all tokens, 
# weighted by the attention mask to ignore padding, and is a common choice in sentence transformers
# (e.g., Sentence-BERT) because it often outperforms using the [CLS] token alone for sentence-level tasks.

# I didn’t add a projection layer after pooling to keep the architecture simple and preserve the 
# 768-dimensional embeddings from BERT, which are already rich and usable for downstream tasks.