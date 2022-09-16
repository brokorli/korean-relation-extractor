import os
from ..io import load_pkl_file, load_index, download_file_from_hf_hub


FAISS_INDEX_DUMP = "faiss_index.dump"
INDEX_RELATION_PAIRS = "index_relation_pairs.dump"


class DataStore:
    def __init__(self):
        self.load()

    def load(self):
        self.faiss_index = load_index(
            path=download_file_from_hf_hub(
                repo_id=os.environ["BROKORLI_RE_REPO_NAME"], file_name=FAISS_INDEX_DUMP
            )
        )

        self.index_relation_pairs = load_pkl_file(
            path=download_file_from_hf_hub(
                repo_id=os.environ["BROKORLI_RE_REPO_NAME"],
                file_name=INDEX_RELATION_PAIRS,
            )
        )

    def search(self, hidden_state, top_k: int):
        return self.faiss_index.search(hidden_state, top_k)

    def get_label_from_index(self, index: int):
        return self.index_relation_pairs[index]
