import os
from .data import RelationData
from .data_store import DataStore
from .model import ReModel
from .io import load_txt_file, download_file_from_hf_hub


class BrokorliRE:
    def __init__(self, top_k: int = 12, logit_ratio=0.3):
        self.data_store = DataStore()
        self.model = ReModel(
            data_store=self.data_store, top_k=top_k, logit_ratio=logit_ratio
        )

        self.relations = load_txt_file(
            path=download_file_from_hf_hub(
                repo_id=os.environ["BROKORLI_RE_REPO_NAME"], file_name="relations.txt"
            )
        )

        self.relation_label_map = {
            f"[LABEL{i + 1}]": self.relations[i] for i in range(len(self.relations))
        }

    def extract(self, sentence: str, subj: str, obj: str):
        output = self.model(sentence=sentence, subj=subj, obj=obj)

        return RelationData(
            sentence=sentence,
            subj=subj,
            obj=obj,
            relation=self.relation_label_map[output],
        )
