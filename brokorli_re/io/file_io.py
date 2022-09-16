import faiss
import pickle as pkl


def load_txt_file(path):
    buffer = []

    with open(path, "r", encoding="utf-8") as fp:
        for line in fp.readlines():
            line = line.replace("\n", "")

            if line:
                buffer.append(line)

    return buffer


def load_pkl_file(path):
    with open(path, "rb") as fp:
        return pkl.load(fp)


def load_index(path):
    return faiss.read_index(path)
