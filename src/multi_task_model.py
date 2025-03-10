# src/multi_task_model.py
import torch
import torch.nn as nn
from sentence_transformer import SentenceTransformer
from constants import sentences


class MultiTaskModel(nn.Module):
    """
    Multi-task model with a shared SentenceTransformer backbone.
    Task A: Sentence Classification (Negative, Neutral, Positive).
    Task B: NER (Person, Organization, Location, Date) - simplified to sentence-level.
    """
    def __init__(self, transformer_model):
        super(MultiTaskModel, self).__init__()
        self.transformer = transformer_model
        self.embedding_dim = transformer_model.embedding_dim
        # Task A: Sentiment classification (3 classes)
        self.classifier = nn.Linear(self.embedding_dim, 3)
        # Task B: NER (4 entity types)
        self.ner_head = nn.Linear(self.embedding_dim, 4)

    def forward(self, in_sentences):
        """Forward pass for multi-task learning."""
        embeddings = self.transformer(in_sentences)
        class_logits = self.classifier(embeddings)  # Sentiment logits
        ner_logits = self.ner_head(embeddings)     # NER logits
        return class_logits, ner_logits


def test_multi_task_model():
    """Test the multi-task model with sample sentences."""
    base_transformer = SentenceTransformer()
    model = MultiTaskModel(base_transformer)
    model.eval()
    class_logits, ner_logits = model(sentences)
    print(f"Class logits shape: {class_logits.shape}")
    print(f"NER logits shape: {ner_logits.shape}")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Class logits: {class_logits[i].tolist()}")
        print(f"NER logits: {ner_logits[i].tolist()}")


if __name__ == "__main__":
    test_multi_task_model()

# Task 2 Explanation:
# - Shared Backbone: Reused SentenceTransformer for feature extraction.
# - Task A: Sentiment (3 classes) - simple linear head for classification.
# - Task B: NER (4 classes) - simplified to sentence-level tagging for demo purposes.
# - Architecture: Independent heads on shared embeddings, no task interaction.
