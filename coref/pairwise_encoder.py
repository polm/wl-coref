""" Describes PairwiseEncodes, that transforms pairwise features, such as
distance between the mentions, same/different speaker into feature embeddings
"""

import torch

from coref.const import Doc


class DistancePairwiseEncoder(torch.nn.Module):
    
    def __init__(self, embedding_size, dropout_rate):
        super().__init__()
        emb_size = embedding_size
        self.distance_emb = torch.nn.Embedding(9, emb_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.shape = emb_size

    @property
    def device(self) -> torch.device:
        """ A workaround to get current device (which is assumed to be the
        device of the first parameter of one of the submodules) """
        return next(self.distance_emb.parameters()).device
    

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                top_indices: torch.Tensor,
                doc: Doc) -> torch.Tensor:
        word_ids = torch.arange(0, len(doc["cased_words"]), device=self.device)
        distance = (word_ids.unsqueeze(1) - word_ids[top_indices]
                    ).clamp_min_(min=1)
        log_distance = distance.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=6).to(torch.long)
        distance = torch.where(distance < 5, distance - 1, log_distance + 2)
        distance = self.distance_emb(distance)
        return self.dropout(distance)
