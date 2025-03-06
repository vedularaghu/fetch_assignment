# Training Considerations

## 1. Entire Network Frozen

### Implications
- All parameters (BERT backbone and task heads) are frozen, so no learning occurs because gradients are not computed.
- Model outputs depend solely on pre-trained or random weights.

### Advantages
- None for training; may be useful for inference with a fully pre-trained model, but not practical for task-specific training.

### Training Approach
- Training is not feasible; requires pre-trained weights for both tasks.

## 2. Only Transformer Backbone Frozen

### Implications
- BERT's weights are fixed.
- Only the task-specific heads (sentence_classifier and ner_classifier) are trained.
- Reduces trainable parameters from ~110M to ~15K (depending on head sizes).

### Advantages
- Faster training due to fewer parameter updates.
- Prevents overfitting when task-specific data is limited by leveraging pre-trained features.
- Maintains general language understanding from pre-training.

### Disadvantages
- Limits adaptation of the backbone to task-specific needs, potentially capping performance if fine-tuned shared representations are needed.

### Training Approach
- Use a standard optimizer (e.g., AdamW) to update only the head parameters.
- Set `requires_grad=False` for BERT layers.

## 3. Only One Task-Specific Head Frozen

### Implications
- Assume the sentence classification head is frozen.
- The NER head and possibly the backbone are trained.

#### Sub-Case A: Backbone Not Frozen

**Implications:**
- Backbone and NER head update, but sentence classification performance may degrade as the shared backbone changes.

**Advantages:**
- Useful for adapting the model to a new task (NER) while preserving a pre-trained task (sentence classification), if the latter is already optimized.

**Training Approach:**
- Freeze the sentence head.
- Fine-tune the backbone and NER head, possibly with a lower learning rate for the backbone (e.g., 2e-5 vs. 1e-3 for the head).

#### Sub-Case B: Backbone Frozen

**Implications:**
- Only the NER head is trained, resembling single-task training on a frozen backbone.

**Advantages:**
- Preserves both the pre-trained backbone and sentence classification performance.

**Training Approach:**
- Freeze both BERT and the sentence head.
- Train only the NER head.

**Disadvantages:**
- Fine-tuning the backbone (as in Sub-Case A) risks "catastrophic forgetting" for the frozen task unless mitigated (e.g., with regularization).

## Transfer Learning Strategy

### Pre-trained Model
- Use `bert-base-uncased`, pre-trained on masked language modeling and next sentence prediction, aligning well with both sentence classification and NER tasks.

### Layers to Freeze/Unfreeze
- **Limited Data:** Freeze the BERT backbone and train only the task heads to leverage pre-trained features and prevent overfitting.
- **Sufficient Data:** Fine-tune the entire model (backbone + heads) to adapt BERT's representations to both tasks, enhancing shared learning in multi-task settings.

### Rationale
- BERT's pre-training provides robust general language understanding for NLP tasks.
- Freezing the backbone is ideal for small datasets (<10K samples) to prevent overfitting.
- Fine-tuning the backbone is beneficial with larger datasets, allowing for task-specific optimization.
- In multi-task learning, fine-tuning the backbone can improve performance by balancing representations for both tasks.
