import torch.nn as nn
import warnings
from config.load_logger import console_logger

import torch

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
        console_logger.info(
            f"Dual encoder model for Q&A successfully initialized with:\n\n embedding dimension:{embed_dim} \n "
            f"output_dim:{output_dim} ")

    def forward(self, question, answer):
        q_embeddings = self.q_encoder(question)
        a_embeddings = self.a_encoder(answer)
        return q_embeddings, a_embeddings, q_embeddings @ a_embeddings.T
