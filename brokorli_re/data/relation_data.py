from dataclasses import dataclass


@dataclass
class RelationData:
    sentence: str
    subj: str
    obj: str
    relation: str
