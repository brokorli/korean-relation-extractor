import torch
from ..data_store import DataStore


class ReModelOutput:
    def __init__(
        self,
        device: str,
        data_store: DataStore,
        vocab_size: int,
        mask_token_id: int,
        top_k: int,
        logit_ratio: float,
    ):
        self.device = device
        self.data_store = data_store
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.top_k = top_k
        self.logit_ratio = logit_ratio

    def __call__(self, input_ids, logits, hidden_state):
        mask_logits = self.get_mask_logits(
            logits=logits,
            mask_idxes=(input_ids == self.mask_token_id).nonzero(as_tuple=True)[1],
        )

        knn_logits = self.get_knn_logits(
            hidden_state=self.get_mask_hidden_state(
                hidden_state=hidden_state,
                mask_idxes=(input_ids == self.mask_token_id).nonzero(as_tuple=True)[1],
            )
        )

        logits = self.logit_ratio * knn_logits + (1 - self.logit_ratio) * mask_logits

        return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)

    def get_mask_logits(self, logits, mask_idxes):
        return logits[torch.arange(logits.shape[0]), mask_idxes]

    def get_mask_hidden_state(self, hidden_state, mask_idxes):
        return hidden_state[torch.arange(hidden_state.shape[0]), mask_idxes]

    def get_knn_logits(self, hidden_state):
        knn_logits = torch.full((hidden_state.shape[0], self.vocab_size), 1000.0).to(
            self.device
        )

        distances, indexes = self.data_store.search(
            hidden_state=hidden_state.cpu().numpy(), top_k=self.top_k
        )
        distances = torch.from_numpy(distances).to(self.device)

        for i in range(hidden_state.shape[0]):
            for j in range(self.top_k):
                if (
                    knn_logits[i][self.data_store.get_label_from_index(indexes[i][j])]
                    > distances[i][j]
                ):
                    knn_logits[i][
                        self.data_store.get_label_from_index(indexes[i][j])
                    ] = distances[i][j]

        if torch.sum(knn_logits) != 0.0:
            knn_logits = torch.softmax((-1) * knn_logits, dim=-1)

        return knn_logits
