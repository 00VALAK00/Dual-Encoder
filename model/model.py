import torch.nn as nn
from transformers import AutoTokenizer
import tqdm
import warnings
import logging
import torch
from loss.loss import CrossEntropyLoss
import numpy as np

warnings.filterwarnings("ignore")


class Encoder(nn.Module):
    def __init__(self, tokenizer, embed_dim, output_dim):
        super(Encoder, self).__init__()
        vocab_size = tokenizer.vocab_size
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim,
                                       nhead=8,
                                       batch_first=True
                                       ),
            norm=nn.LayerNorm(embed_dim),
            num_layers=3,
            enable_nested_tensor=False
        )
        self.projection_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, tokenizer_output):
        x = self.embed_layer(tokenizer_output["input_ids"])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output["attention_mask"].logical_not())
        cls = x[:, 0, :]
        return self.projection_layer(cls)


class DualEncoder(torch.nn.Module):

    def __init__(self, tokenizer, embed_dim, output_dim):
        super(DualEncoder, self).__init__()
        self.q_encoder = Encoder(tokenizer, embed_dim, output_dim)
        self.a_encoder = Encoder(tokenizer, embed_dim, output_dim)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, question, answer):
        q_embeddings = self.q_encoder(question)
        a_embeddings = self.a_encoder(answer)
        return q_embeddings, a_embeddings, q_embeddings @ a_embeddings.T

    def train(self, epochs=20, criterion=CrossEntropyLoss(), data_loader=None):
        num_batches = len(data_loader)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            running_loss = []

            for batch, (q, a) in enumerate(tqdm.tqdm(data_loader)):

                q = {k: v.to(self.device) for k, v in q.items()}
                a = {k: v.to(self.device) for k, v in a.items()}

                optimizer.zero_grad()

                _, _, similarity_scores = self(q, a)
                if torch.isnan(similarity_scores).any():
                    raise ValueError("Nan values detected in similarity_scores.")
                loss = criterion(similarity_scores)

                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if batch % 10 == 0:
                    print(f"Epoch {epoch}, loss = {np.mean(running_loss)}")
