# %%
import string
import re
from typing import List


def tokenizer_with_preprocessing(text: str):
    text = re.sub("<br />", "", text)

    assert string.punctuation == "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in string.punctuation:
        if p not in [".", ","]:
            text = text.replace(p, " ")

    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text.strip().split()


if __name__ == "__main__":
    assert tokenizer_with_preprocessing("I like cats.") == ["I", "like", "cats", "."]
