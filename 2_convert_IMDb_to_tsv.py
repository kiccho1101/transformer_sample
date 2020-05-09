# %%
import glob
import os
import io
import string


def convert_to_tsv():
    """
    pos -> 1
    neg -> 0
    というふうにラベル付けしてtsvファイルに保存する
    """
    with open("./data/IMDb_train.tsv", "w") as f:
        for fname in glob.glob(os.path.join("./data/aclImdb/train/pos/", "*.txt")):
            with io.open(fname, "r", encoding="utf-8") as ff:
                text = ff.readline()
                text = text.replace("\t", " ")
                text = text + "\t" + "1" + "\t" + "\n"
                f.write(text)

        for fname in glob.glob(os.path.join("./data/aclImdb/train/neg/", "*.txt")):
            with io.open(fname, "r", encoding="utf-8") as ff:
                text = ff.readline()
                text = text.replace("\t", " ")
                text = text + "\t" + "0" + "\t" + "$0"
                f.write(text)

    with open("./data/IMDb_test.tsv", "w") as f:
        for fname in glob.glob(os.path.join("./data/aclImdb/test/pos/", "*.txt")):
            with io.open(fname, "r", encoding="utf-8") as ff:
                text = ff.readline()
                text = text.replace("\t", " ")
                text = text + "\t" + "1" + "\t" + "\n"
                f.write(text)

        for fname in glob.glob(os.path.join("./data/aclImdb/test/neg/", "*.txt")):
            with io.open(fname, "r", encoding="utf-8") as ff:
                text = ff.readline()
                text = text.replace("\t", " ")
                text = text + "\t" + "0" + "\t" + "$0"
                f.write(text)


if __name__ == "__main__":
    convert_to_tsv()
