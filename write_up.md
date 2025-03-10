# Write-Up for Tasks 3 and 4

## Task 3: Training Considerations

### Scenarios
1. **Entire Network Frozen**:
   - **Implications**: 
     - No parameters in the model (transformer backbone, classifier, or ner_head) are updated during training. 
     - The model acts solely as a feature extractor, relying entirely on pre-trained weights. 
     - Outputs (logits for sentiment and NER) are fixed based on the initial weights, meaning no task-specific learning occurs.

   - **Advantages**: 
     - Speed: No training is required, making it extremely fast for inference on new data. 
     - Resource Efficiency: Minimal computational cost, ideal for deployment on resource-constrained environments. 
     - Stability: Preserves the pre-trained model’s general knowledge without risk of overfitting or catastrophic forgetting.


   - **Rationale for training**:
     - Since everything is frozen, no training occurs in this scenario. Instead, we would use the model `as-is`:
     - Extract embeddings with the frozen SentenceTransformer. 
     - Pass these embeddings through the frozen classifier and ner_head (per my NLP use case) to get logits. 
     - Evaluate performance on tasks using pre-computed outputs.

    
2. **Transformer Backbone Frozen**:
   - **Implications**:
     - The SentenceTransformer backbone is frozen, while the task-specific heads (classifier and ner_head) are trainable. 
     - Embeddings remain fixed, but the linear layers adapt to optimize task-specific objectives (e.g., sentiment classification and NER). 
     - Gradients only flow through the heads, not the transformer.

   - **Advantages**:
     - Efficiency: Training is faster and less memory-intensive than fine-tuning the entire model, as the transformer’s many parameters (e.g., ~22M for MiniLM) aren’t updated. 
     - Leverages Pre-training: Retains the transformer’s robust, general-purpose embeddings while tailoring the heads to our tasks. 
     - Reduced Overfitting Risk: Fewer trainable parameters lower the chance of overfitting, especially with small datasets.
   - **Rationale for training**:
     - How to Train:
       - Freeze the transformer:
       - ```python
         for param in model.transformer.parameters():
             param.requires_grad = False
         ```

       - Define optimizers only for the heads:
       - ```python
         optimizer = optim.Adam([{'params': model.classifier.parameters()},
                    {'params': model.ner_head.parameters()}], lr=1e-3)
         ```
   - **Use case**: Ideal when the transformer’s embeddings are already well-suited to our domain (e.g., general English text), but the tasks require custom output mappings


3. **One Task Head Frozen (e.g., Classifier)**:
    - **Implications**:
        - The transformer backbone and one head (e.g., ner_head) are trainable, while the other (e.g., classifier) is frozen. 
        - The frozen head’s predictions remain fixed, while the transformer and the trainable head adapt to their task. 
        - Gradients flow through the transformer and the trainable head, potentially influencing the embeddings to favor that task.

   - **Advantages**:
     - Task Prioritization: Allows improvement on one task (e.g., NER) while preserving performance on another (e.g., sentiment) where the pre-trained head is sufficient. 
     - Flexibility: The transformer can still fine-tune to the trainable task’s needs, enhancing shared feature quality. 
     - Data Efficiency: Useful when one task has limited labeled data, leveraging the frozen head’s pre-trained capability.

   - **Rationale for Training**:
     - How to Train:
        - Freeze the classifier head:

        - ```python
          for param in model.classifier.parameters():
              param.requires_grad = False
          ```
       - Optimize the transformer and ner_head:
       - ```python
         optimizer = optim.Adam([{'params': model.transformer.parameters(), 'lr': 2e-5},
         {'params': model.ner_head.parameters(), 'lr': 1e-3)])
         ```

       - Compute loss only for the trainable task (NER) and backpropagate:
       ```python
        ner_loss = ner_loss_fn(ner_logits, batch_ner)
        ner_loss.backward()
        ```
   - **Use case**: If sentiment classification is already well-solved (e.g., via pre-training or prior tuning), but NER needs improvement due to domain-specific entities.



### Transfer Learning
- **Pre-trained Model**: `all-MiniLM-L6-v2` - lightweight and efficient, good for sentence embeddings. Pretrained on a diverse text
- **Freeze/Unfreeze**: Freeze transformer initially, train heads, then optionally unfreeze transformer. Ideally this allows us to leverage the general knowledge of the transformer while fine tuning to task-specific patterns
- **Rationale**: Start with general knowledge, adapt heads to tasks, fine-tune transformer for domain if needed.

### Insights
Freezing strategies balance efficiency and adaptability. Transfer learning leverages MiniLM’s sentence-level pre-training, making it a strong starting point.

## Task 4: Training Loop

### Key Decisions
- **Data**: Chose sentences with clear entities (e.g., "Samson", "Fetch", "London") to simplify NER learning.
- **Loss**: Equal weighting of tasks; could adjust based on task priority.
- **Metrics**: Accuracy for both tasks; F1 could be added for NER.

### Insights
- MTL benefits from shared embeddings, but task imbalance could require loss weighting.
- Simplified NER (sentence-level) works for demo but limits realism; token-level NER would need a different approach.

This setup demonstrates MTL fundamentals while keeping it reproducible and extensible.
