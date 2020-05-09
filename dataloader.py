# %%
import random
import torch
import torchtext
from torchtext.vocab import Vectors

from typing import Tuple

from tokenizer import tokenizer_with_preprocessing


def define_field(
    max_length: int = 256,
) -> Tuple[torchtext.data.Field, torchtext.data.Field]:
    """
    torchtext.data.Fieldを定義する
    torchtext.data.Field: データの種類ごとの前処理とVocabを保持

    max_length: 最大単語数

    .preprocess(): 前処理をおこなう
    .build_vocab(): 引数のデータからvocab作成
    .pad(): paddingされたデータを作成
    .numericalize(): Variableに変換

    参考資料: https://www.slideshare.net/DeepLearningJP2016/dlhacks-lt-pytorchdataloader-torchtext
    """
    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenizer_with_preprocessing,
        use_vocab=True,
        lower=True,
        include_lengths=True,
        batch_first=True,
        fix_length=max_length,
        init_token="<cls>",
        eos_token="<eos>",
    )
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    assert TEXT.preprocess("I like cats.") == ["i", "like", "cats", "."]
    return TEXT, LABEL


def read_dataset(
    TEXT: torchtext.data.Field, LABEL: torchtext.data.Field
) -> Tuple[torchtext.data.TabularDataset, torchtext.data.TabularDataset]:
    """
    tsvファイルからTabularDataset型でデータを読み込む
    """
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path="./data/",
        train="IMDb_train.tsv",
        test="IMDb_test.tsv",
        format="tsv",
        fields=[("Text", TEXT), ("Label", LABEL)],
    )
    assert len(train_val_ds) == 12501
    assert list(vars(train_val_ds[0]).keys()) == ["Text", "Label"]
    return train_val_ds, test_ds


def split_train_ds(
    train_val_ds: torchtext.data.TabularDataset,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    train_val_ds -> train_ds, val_ds
    に分割する
    """
    train_ds: torch.utils.data.Dataset
    val_ds: torch.utils.data.Dataset
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(42))
    assert len(train_ds) == 10001
    assert len(val_ds) == 2500
    assert list(vars(train_ds[0]).keys()) == ["Text", "Label"]
    assert list(vars(val_ds[0]).keys()) == ["Text", "Label"]
    return train_ds, val_ds


def build_vocab(
    TEXT: torchtext.data.Field, train_ds: torch.utils.data.Dataset
) -> Tuple[torchtext.data.Field, torchtext.vocab.Vectors]:
    """
    (単語, ベクトル)
    のリスト(vocab)を作成する。

    fasttextのpre-trainedベクトルを利用して、train_ds内に出現する単語のvocabを作成する
    """
    english_fasttext_vectors = Vectors(name="data/wiki-news-300d-1M.vec")
    assert english_fasttext_vectors.dim == 300
    assert len(english_fasttext_vectors.itos) == 999994
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)
    assert TEXT.vocab.vectors.shape == torch.Size([12268, 300])
    assert len(TEXT.vocab.itos) == 12268
    return TEXT, english_fasttext_vectors


def create_dataloader(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    test_ds: torchtext.data.TabularDataset,
    batch_size: int = 24,
    max_length: int = 256,
) -> Tuple[torchtext.data.Iterator, torchtext.data.Iterator, torchtext.data.Iterator]:
    """
    datasetからdataloaderを作成
    """
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False
    )
    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False
    )
    assert next(iter(val_dl)).Text[0].shape == torch.Size([batch_size, max_length])
    assert next(iter(val_dl)).Text[1].shape == torch.Size([batch_size])
    assert next(iter(val_dl)).Label.shape == torch.Size([batch_size])
    return train_dl, val_dl, test_dl


def get_IMDb_Dataloaders_and_TEXT(
    max_length: int = 256, batch_size: int = 24
) -> Tuple[
    torchtext.data.Iterator,
    torchtext.data.Iterator,
    torchtext.data.Iterator,
    torchtext.data.Field,
]:
    TEXT, LABEL = define_field(max_length)
    train_val_ds, test_ds = read_dataset(TEXT, LABEL)
    train_ds, val_ds = split_train_ds(train_val_ds)
    TEXT, english_fasttext_vectors = build_vocab(TEXT, train_ds)
    train_dl, val_dl, test_dl = create_dataloader(
        train_ds, val_ds, test_ds, batch_size=batch_size, max_length=max_length
    )
    return train_dl, val_dl, test_dl, TEXT


if __name__ == "__main__":
    get_IMDb_Dataloaders_and_TEXT()
