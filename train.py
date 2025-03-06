import torch.nn.functional as F
from torch.optim import AdamW
from multi_task_model import MultiTaskModel
from torch.utils.data import DataLoader, Dataset

# Initialize model and optimizer
model = MultiTaskModel(num_sentence_classes=2, num_ner_classes=5)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Hypothetical training loop
def train_epoch(model, dataloader):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids']  # Shape: (batch_size, seq_len)
        attention_mask = batch['attention_mask']  # Shape: (batch_size, seq_len)
        sentence_labels = batch['sentence_labels']  # Shape: (batch_size)
        ner_labels = batch['ner_labels']  # Shape: (batch_size, seq_len), -100 for padding
        
        # Forward pass
        sentence_logits, ner_logits = model(input_ids, attention_mask)
        
        # Compute losses
        sentence_loss = F.cross_entropy(sentence_logits, sentence_labels)
        ner_loss = F.cross_entropy(ner_logits.view(-1, num_ner_classes), 
                                  ner_labels.view(-1), ignore_index=-100)
        total_loss = sentence_loss + ner_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Compute metrics
        sentence_preds = torch.argmax(sentence_logits, dim=1)
        sentence_acc = (sentence_preds == sentence_labels).float().mean()
        ner_preds = torch.argmax(ner_logits, dim=-1)
        ner_mask = (ner_labels != -100)
        ner_acc = (ner_preds[ner_mask] == ner_labels[ner_mask]).float().mean()
        
        print(f"Sentence Loss: {sentence_loss.item():.4f}, NER Loss: {ner_loss.item():.4f}, "
              f"Sentence Acc: {sentence_acc.item():.4f}, NER Acc: {ner_acc.item():.4f}")
