import os
from .brokorli_re import BrokorliRE


os.environ["BROKORLI_RE_REPO_NAME"] = "chnaaam/korean-relation-extraction"

__all__ = [BrokorliRE]