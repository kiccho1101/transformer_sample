# %%
import os
import urllib
import zipfile
import tarfile


def download_fasttext():
    """
    fasttextのpretrained vectorsをダウンロードする。
    """
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    save_path = "./data/wiki-news-300d-1M.vec.zip"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
    zip.extractall("./data/")
    zip.close()


def download_Imdb_data():
    """
    IMDb(映画レビューサイト)のデータセットをダウンロードする。
    """
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./data/aclImdb_v1.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)

    tar = tarfile.open("./data/aclImdb_v1.tar.gz")
    tar.extractall("./data/")
    tar.close()


if __name__ == "__main__":
    download_fasttext()
    download_Imdb_data()
