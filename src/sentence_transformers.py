# src/sentence_transformer.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from constants import sentences


class SentenceTransformer(nn.Module):
    """
    A custom Sentence Transformer model that encodes sentences into fixed-length embeddings.
    Uses a pre-trained transformer backbone with mean pooling.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(SentenceTransformer, self).__init__()
        # Load pre-trained transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = 384  # MiniLM-L6-v2 hidden size
        # Define pooling as a lambda function for flexibility
        self.pooling = lambda x, mask: self.mean_pooling(x, mask)

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling to convert token embeddings to sentence embeddings.
        Respects attention mask to ignore padding tokens.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sentences):
        """Encode input sentences into embeddings."""
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"].to(self.transformer.device)
        attention_mask = encoded_input["attention_mask"].to(self.transformer.device)

        # Get token embeddings from transformer
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        # Apply pooling
        return self.pooling(token_embeddings, attention_mask)


def test_sentence_transformer():
    """Test the SentenceTransformer with sample sentences."""
    model = SentenceTransformer()
    model.eval()
    embeddings = model(sentences)
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Embeddings shape: {embeddings.shape}")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Embedding (first 5 dims): {embeddings[i, :5].tolist()} ...")


if __name__ == "__main__":
    test_sentence_transformer()

# Task 1 Explanation:
# - Backbone: Used 'all-MiniLM-L6-v2' for efficiency (384 dims) and sentence-level performance.
# - Pooling: I chose mean pooling over max or CLS token for simplicity and robustness.
# - No extra layers: I kept it minimal to focus on raw embeddings as per task requirements.
