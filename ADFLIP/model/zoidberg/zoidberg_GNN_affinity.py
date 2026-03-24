import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from model.zoidberg.zoidberg_GNN import Zoidberg_GNN
from model.zoidberg.affinity_head import AffinityHead


class Zoidberg_GNN_Affinity(nn.Module):
    """
    Thin wrapper: add an affinity prediction head on top of Zoidberg_GNN's
    per-residue latent embedding.

    This avoids changing Zoidberg_GNN (so it stays compatible with existing
    ADFLIP checkpoints).
    """

    def __init__(
        self,
        *zoidberg_args,
        affinity_head_hidden_dims: tuple[int, ...] = (256, 128),
        affinity_head_use_lightattn: bool = True,
        affinity_head_lightattn_dropout: float = 0.25,
        **zoidberg_kwargs,
    ):
        super().__init__()
        self.backbone = Zoidberg_GNN(*zoidberg_args, **zoidberg_kwargs)
        self.affinity_head = AffinityHead(
            dim=self.backbone.hidden_dim,
            hidden_dims=affinity_head_hidden_dims,
            use_lightattn=affinity_head_use_lightattn,
            lightattn_dropout=affinity_head_lightattn_dropout,
        )

    def forward(self, batch_dict: dict, timestep: torch.Tensor, return_affinity: bool = False):
        """
        Returns:
          - if return_affinity=False: (flatten_logits, residue_x)
          - if return_affinity=True:  (flatten_logits, residue_x, affinity)
        """
        flatten_logits, residue_x = self.backbone(batch_dict, timestep)
        if not return_affinity:
            return flatten_logits, residue_x

        # residue_x is on center residues, densely batched by batch_index of centers.
        center_is_protein = batch_dict["is_protein"][batch_dict["is_center"]]
        center_batch_idx = batch_dict["batch_index"][batch_dict["is_center"]]
        if center_batch_idx.max() + 1 != batch_dict["residue_token"].size(0):
            center_batch_idx = center_batch_idx - center_batch_idx.min()

        protein_mask, _ = to_dense_batch(center_is_protein.bool(), center_batch_idx)
        affinity = self.affinity_head(residue_x, protein_mask)
        return flatten_logits, residue_x, affinity

