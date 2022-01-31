""" Describes the loss function used to train the model, which is a weighted
sum of NLML and BCE losses. """

import torch
from coref.spacy_util import _get_ground_truth

class CorefLoss(torch.nn.Module):
    """ See the rationale for using NLML in Lee et al. 2017
    https://www.aclweb.org/anthology/D17-1018/
    The added weighted summand of BCE helps the model learn even after
    converging on the NLML task. """

    def __init__(self, bce_weight: float):
        assert 0 <= bce_weight <= 1
        super().__init__()
        self._bce_module = torch.nn.BCEWithLogitsLoss()
        self._bce_weight = bce_weight

    def forward(
        self,
        cluster_ids: torch.Tensor,
        coref_scores: torch.Tensor,
        coref_indices: torch.Tensor
    ) -> torch.Tensor:
        """ Returns a weighted sum of two losses as a torch.Tensor """
        cluster_ids = torch.tensor(cluster_ids).to(coref_scores.device)
        pair_mask = torch.arange(coref_scores.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(coref_scores.device)
        pair_mask = pair_mask[:, :coref_scores.size(1) - 1]
        target = _get_ground_truth(
            cluster_ids,
            coref_indices,
            (pair_mask > float("-inf")
             )
        )
        return (self._nlml(coref_scores, target)
                + self._bce(coref_scores, target) * self._bce_weight)

    def _bce(self,
             input_: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
        """ For numerical stability, clamps the input before passing it to BCE.
        """
        return self._bce_module(torch.clamp(input_, min=-50, max=50), target)

    @staticmethod
    def _nlml(input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gold = torch.logsumexp(input_ + torch.log(target), dim=1)
        input_ = torch.logsumexp(input_, dim=1)
        return (input_ - gold).mean()
