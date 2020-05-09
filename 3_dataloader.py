# %%
import torchtext
from tokenizer import tokenizer_with_preprocessing

max_length = 256
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


# %%
train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path="./data/",
    train="IMDb_train.tsv",
    test="IMDb_test.tsv",
    format="tsv",
    fields=[("Text", TEXT), ("Label", LABEL)],
)

# %%
train_val_ds.fields["Text"].sequential
