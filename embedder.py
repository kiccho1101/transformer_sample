# %%
import torch
import torch.nn as nn
from dataloader import get_IMDb_Dataloaders_and_TEXT


class Embedder(nn.Module):
    """idで示されている単語をベクトルに変換する"""

    def __init__(self, text_embedding_vectors: torch.Tensor):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True
        )

    def forward(self, x) -> nn.Embedding:
        x_vec = self.embeddings(x)
        return x_vec


if __name__ == "__main__":
    train_dl, val_dl, test_dl, TEXT = get_IMDb_Dataloaders_and_TEXT(
        max_length=256, batch_size=24
    )
    batch = next(iter(train_dl))
    net1 = Embedder(TEXT.vocab.vectors)

    x_in = batch.Text[0]
    x_out = net1(x_in)

    # %%
    assert x_in.shape == torch.Size([24, 256])
    assert x_out.shape == torch.Size([24, 256, 300])
