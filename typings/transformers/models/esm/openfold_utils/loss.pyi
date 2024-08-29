"""
This type stub file was generated by pyright.
"""

import torch
from typing import Dict, Optional

def compute_predicted_aligned_error(logits: torch.Tensor, max_bin: int = ..., no_bins: int = ..., **kwargs) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    ...

def compute_tm(logits: torch.Tensor, residue_weights: Optional[torch.Tensor] = ..., max_bin: int = ..., no_bins: int = ..., eps: float = ..., **kwargs) -> torch.Tensor:
    ...
