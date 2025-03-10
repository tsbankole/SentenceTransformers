import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from multi_task_model import MultiTaskModel
from sentence_transformer import SentenceTransformer
from constants import sentences


def train_multi_task_model(model, epochs=3, batch_size=2):
    """
    Training loop for the multi-task model.
    Handles hypothetical data, forward pass, and metrics.
    """
    # Hypothetical data with clear entities for NER
    class_labels = torch.tensor([2, 2, 0, 1])  # Pos, Pos, Neg, Neu
    ner_labels = torch.tensor([0, 1, 2, 3])  # Person, Org, Loc, Date

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    class_loss_fn = CrossEntropyLoss()
    ner_loss_fn = CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_class_preds, all_ner_preds = [], []
        all_class_labels, all_ner_labels = [], []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_class = class_labels[i:i + batch_size]
            batch_ner = ner_labels[i:i + batch_size]

            # Forward pass
            class_logits, ner_logits = model(batch_sentences)
            class_loss = class_loss_fn(class_logits, batch_class)
            ner_loss = ner_loss_fn(ner_logits, batch_ner)
            combined_loss = class_loss + ner_loss  # Equal weighting

            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            total_loss += combined_loss.item()

            # Metrics accumulation
            with torch.no_grad():
                all_class_preds.extend(torch.argmax(class_logits, dim=1).tolist())
                all_ner_preds.extend(torch.argmax(ner_logits, dim=1).tolist())
                all_class_labels.extend(batch_class.tolist())
                all_ner_labels.extend(batch_ner.tolist())

        class_acc = (torch.tensor(all_class_preds) == torch.tensor(all_class_labels)).float().mean().item()
        ner_acc = (torch.tensor(all_ner_preds) == torch.tensor(all_ner_labels)).float().mean().item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
        print(f"Class Accuracy: {class_acc:.4f}, NER Accuracy: {ner_acc:.4f}")


if __name__ == "__main__":
    base_transformer = SentenceTransformer()
    model = MultiTaskModel(base_transformer)
    train_multi_task_model(model)

# Task 4 Explanation:
# - Data: Hypothetical sentences with clear entities to aid NER learning.
# - Loss: Summed cross-entropy losses for both tasks (equal weighting).
# - Metrics: Accuracy for simplicity; computed over all samples per epoch.
# - Assumptions: Small dataset fits in memory; no real training, just hypothetical data.
