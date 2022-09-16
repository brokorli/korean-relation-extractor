import os
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .re_model_output import ReModelOutput
from ..data_store import DataStore


PROMPT_TEMPLATE = "[X] [SEP] [E1] [SUBJECT] [/E1] [MASK] [E2] [OBJECT] [/E2]"


class ReModel:
    def __init__(self, data_store: DataStore, top_k=12, logit_ratio=0.3):
        repo_name = os.environ["BROKORLI_RE_REPO_NAME"]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(repo_name)
        self.model = (
            AutoModelForMaskedLM.from_pretrained(repo_name).eval().to(self.device)
        )

        self.model_output = ReModelOutput(
            device=self.device,
            data_store=data_store,
            vocab_size=len(self.tokenizer),
            mask_token_id=self.tokenizer.mask_token_id,
            top_k=top_k,
            logit_ratio=logit_ratio,
        )

    def __call__(self, sentence: str, subj: str, obj: str):
        prompt = self.make_prompt(sentence=sentence, subj=subj, obj=obj)

        inputs = self.tokenize(prompt=prompt)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            logits = outputs.logits
            hidden_state = outputs.hidden_states[-1]

            output = self.model_output(
                input_ids=inputs["input_ids"], logits=logits, hidden_state=hidden_state
            )

            return self.tokenizer.convert_ids_to_tokens(output.tolist()[0])

    @staticmethod
    def make_prompt(sentence: str, subj: str, obj: str):
        prompt = PROMPT_TEMPLATE

        for origin_str, replace_str in [
            ("[X]", sentence),
            ("[SUBJECT]", subj),
            ("[OBJECT]", obj),
        ]:
            prompt = prompt.replace(origin_str, replace_str)

        return prompt

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def get_relation_labels(num_labels):
        return [f"[LABEL{i}]" for i in range(1, num_labels + 1)]
