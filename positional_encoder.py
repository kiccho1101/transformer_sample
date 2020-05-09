# %%
import torch
import torch.nn as nn
import math
from dataloader import get_IMDb_Dataloaders_and_TEXT
from embedder import Embedder
from utils import timer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoder(nn.Module):
    """入力された単語の位置を示すベクトル情報を付加する"""

    def __init__(self, d_model: int = 300, max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


if __name__ == "__main__":
    train_dl, val_dl, test_dl, TEXT = get_IMDb_Dataloaders_and_TEXT(
        max_length=256, batch_size=24
    )
    batch = next(iter(train_dl))

    net1 = Embedder(TEXT.vocab.vectors)
    net2 = PositionalEncoder(d_model=300, max_seq_len=256)

    with timer("forward"):
        x = batch.Text[0]
        x1 = net1(x)
        x1 = x1.to(device)
        x2 = net2(x1)

    assert x1.shape == torch.Size([24, 256, 300])
    assert x2.shape == torch.Size([24, 256, 300])
