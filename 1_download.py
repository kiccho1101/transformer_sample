# %%
import os
import urllib
import zipfile
import tarfile

# Download fastText pre-trained model(english)
url = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
)
save_path = "./data/wiki-news-300d-1M.vec.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)


zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
zip.extractall("./data/")
zip.close()


# Download IMDb data
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)

tar = tarfile.open("./data/aclImdb_v1.tar.gz")
tar.extractall("./data/")
tar.close()
