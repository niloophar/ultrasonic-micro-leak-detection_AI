"""
Contrastive losses for the NoFlow-vs-Flow encoder.

Three losses are provided and selected via the ``contrastive_loss`` config:

- :class:`SupConLoss` (Khosla et al., NeurIPS 2020) — class-based. Pulls
  together samples with the same discrete label, pushes apart samples with
  different labels. In this project NoFlow patches across all series cluster
  together (drift invariance) while Flow is pushed away.

- :class:`RNCLoss` (Rank-N-Contrast, Zha et al., NeurIPS 2023) — regression-
  based. Ranks samples by their distance in a *continuous* target space and
  orders the embedding accordingly. In this project we use per-series mean
  ``temp_pipe`` as the continuous target so the latent explicitly encodes
  the seasonal temperature continuum — the direction most strongly linked
  to PC2 drift in the 8-series analysis.

- :class:`DeepSADLoss` (Ruff et al., 2018/2020 Deep SVDD / Deep SAD) —
  asymmetric one-class loss. NoFlow patches are pulled toward a frozen center
  ``c`` on the L2-unit sphere; leakage patches must fall outside a margin
  sphere around ``c`` but are not clustered. Structurally aligns training
  with the downstream GMM (one-class on NoFlow latents). Center ``c`` is
  computed once before training via :meth:`DeepSADLoss.set_center`, then
  frozen for the entire training duration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature: Softmax temperature. Typical values 0.07–0.2.
                     Lower = sharper decision boundary but less stable.

    Expected inputs to ``forward``:
        features: (B, D) L2-normalised projection vectors.
        labels:   (B,)   integer class labels.

    Returns a scalar loss, the mean over all valid anchors.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.zeros((), device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)

        # Mask[i, j] = 1 iff i != j and labels match.
        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask  # exclude self-pairs from positive mask

        # Similarity matrix (features are already L2-normalised).
        logits = torch.matmul(features, features.T) / self.temperature

        # Stable logsumexp trick: subtract max per row before exp.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Exclude self-similarity from the denominator as well.
        logits_mask = 1.0 - self_mask  # 1 everywhere except diagonal
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # For each anchor, mean log-prob over its positives.
        pos_per_anchor = mask.sum(dim=1)
        valid = pos_per_anchor > 0  # anchors with at least one positive

        if valid.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)

        mean_log_prob_pos = (
            (mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]
        )

        return -mean_log_prob_pos.mean()


class RNCLoss(nn.Module):
    """
    Rank-N-Contrast loss (Zha et al., NeurIPS 2023).

    Given features ``v_i`` and *continuous* targets ``y_i``, enforces
    that feature similarity is monotonically ordered by label distance:
    samples with nearly identical ``y`` end up close in feature space,
    samples with distant ``y`` end up far apart — all along a smooth
    manifold rather than in discrete class clusters.

    For each pair (i, j), the normalised likelihood is::

        P(v_j | v_i, S_{i,j}) = exp(sim(v_i, v_j)/τ) / Σ_{k ∈ S_{i,j}} exp(sim(v_i, v_k)/τ)

    where ``S_{i,j} = { k ≠ i : |y_i − y_k| ≥ |y_i − y_j| }`` — everything
    at equal-or-greater label-distance than j. The loss is the mean of
    ``−log P`` over all ordered pairs. See paper Eq. (1)–(2).

    Args:
        temperature: softmax temperature for feature-similarity logits.
                     The paper uses τ=2 for age/temperature regression. Larger
                     τ than SupCon because cosine similarities are already
                     compressed in [−1, 1].
        similarity:  ``"cosine"`` (requires L2-normalised features — matches
                     the existing ``ProjectionHead``) or ``"l2"`` for
                     negative squared L2 distance on un-normalised features.

    Expected inputs to :meth:`forward`:
        features: ``(B, D)`` projection vectors (L2-normalised if
                  ``similarity='cosine'``).
        targets:  ``(B,)`` continuous target values (float tensor).

    Returns a scalar loss averaged over all non-self pairs.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        similarity: str = 'cosine',
    ):
        super().__init__()
        self.temperature = temperature
        if similarity not in ('cosine', 'l2'):
            raise ValueError(f"similarity must be 'cosine' or 'l2', got {similarity!r}")
        self.similarity = similarity

    def forward(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.zeros((), device=device, requires_grad=True)

        targets = targets.float().view(-1, 1)

        # Label distances (L1 on the 1D target).
        label_dist = torch.abs(targets - targets.T)  # (B, B)

        # Feature similarity (higher = more similar).
        if self.similarity == 'cosine':
            sim = torch.matmul(features, features.T)
        else:
            sq_dist = torch.cdist(features, features, p=2) ** 2
            sim = -sq_dist

        logits = sim / self.temperature

        # Stable logsumexp shift per row.
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Exclude self pairs everywhere (k != i).
        self_mask = torch.eye(batch_size, device=device)
        non_self = 1.0 - self_mask

        # diff_mask[i, j, k] = 1 iff label_dist[i, k] >= label_dist[i, j] and k != i.
        # Shape: (B, B, B). For B=256 this is 16M booleans — fine on GPU.
        a = label_dist.unsqueeze(1)   # (B, 1, B)  — i, _, k
        b = label_dist.unsqueeze(2)   # (B, B, 1)  — i, j, _
        diff_mask = (a >= b).float() * non_self.unsqueeze(1)

        # Per-anchor exp logits, excluding self.
        exp_logits = torch.exp(logits) * non_self  # (B, B)

        # For each (i, j), denom[i, j] = Σ_k exp_logits[i, k] * diff_mask[i, j, k].
        denom = (exp_logits.unsqueeze(1) * diff_mask).sum(dim=2)  # (B, B)

        log_prob = logits - torch.log(denom + 1e-12)  # (B, B)

        # Average over all non-self (i, j) pairs.
        n_pairs = non_self.sum()
        if n_pairs == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return -(log_prob * non_self).sum() / n_pairs


class DeepSADLoss(nn.Module):
    """
    Deep SAD asymmetric one-class loss (Ruff et al., 2018/2020).

    NoFlow patches are attracted toward a frozen center ``c`` on the L2-unit
    sphere. Leakage patches are repelled outside a margin sphere but are NOT
    clustered among themselves. Loss formula::

        d²_i = || z_proj_i − c ||²
        L = mean_{NoFlow} d²_i  +  repulsion_weight · mean_{Leakage} max(0, m² − d²_i)

    The center ``c`` is computed once from all train-NoFlow projections via
    :meth:`set_center` before the training loop, then frozen. Call
    :meth:`set_center` before :meth:`forward` — otherwise ``forward`` raises.

    Args:
        margin:            Repulsion margin ``m``. Leakage patches must fall
                           outside the sphere of radius ``m`` around ``c``.
                           With L2-normalised ``z_proj``, distances lie in
                           ``[0, 4]`` (max cosine distance = 2), so ``m`` in
                           ``[0.3, 1.5]`` is a reasonable search range.
        repulsion_weight:  Inner weight ``λ`` on the repulsion term. Tuning
                           ``lambda_deepsad`` (outer weight in total loss)
                           alongside this gives a two-level scale.
    """

    def __init__(self, margin: float = 1.0, repulsion_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.repulsion_weight = repulsion_weight
        self.register_buffer('center', None)

    def set_center(self, embeddings_noflow: torch.Tensor) -> None:
        """Compute mean of NoFlow projections, L2-normalize, freeze as buffer.

        Must be called exactly once, before training starts. Re-initialization
        (e.g., for every new Optuna trial) is supported — call again.

        Args:
            embeddings_noflow: ``(N, D)`` L2-normalised projection vectors for
                               all train-NoFlow patches. Collected via a full
                               deterministic forward pass before the epoch loop.
        """
        c = embeddings_noflow.mean(dim=0)
        c = F.normalize(c, dim=0)
        self.center = c.detach().clone()

    def forward(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute asymmetric one-class loss.

        Args:
            z:      ``(B, D)`` L2-normalised projection vectors.
            labels: ``(B,)`` integer labels. 0 = NoFlow (attract), 1 = Flow (repel).
        """
        if self.center is None:
            raise RuntimeError(
                "DeepSADLoss.forward called before set_center. "
                "Call set_center() with train-NoFlow projections before training."
            )
        device = z.device
        c = self.center.to(device)

        d2 = ((z - c) ** 2).sum(dim=1)  # (B,) squared distances

        noflow_mask = labels == 0
        flow_mask = labels == 1

        if noflow_mask.any():
            l_attract = d2[noflow_mask].mean()
        else:
            l_attract = torch.zeros((), device=device)

        if flow_mask.any():
            l_repulse = torch.clamp(self.margin ** 2 - d2[flow_mask], min=0.0).mean()
        else:
            l_repulse = torch.zeros((), device=device)

        return l_attract + self.repulsion_weight * l_repulse


def build_contrastive_criterion(
    loss_type: str,
    supcon_temperature: float = 0.1,
    rnc_temperature: float = 2.0,
    rnc_similarity: str = 'cosine',
    deepsad_margin: float = 1.0,
    deepsad_repulsion_weight: float = 1.0,
):
    """Factory for the contrastive loss component selected via config.

    Returns a dict ``{"supcon": ...|None, "rnc": ...|None, "deepsad": ...|None}``.
    Training code checks which keys are not ``None`` and feeds the appropriate
    targets (discrete labels vs continuous targets).

    Accepted loss types:
      - ``"supcon"`` (default): class-based, uses discrete labels.
      - ``"rnc"``: regression-based, uses continuous targets.
      - ``"both"`` / ``"supcon+rnc"``: both losses, weighted separately.
      - ``"deepsad"``: asymmetric one-class (mutually exclusive with supcon/rnc).
      - ``"none"``: disable contrastive loss entirely (auxiliary head only).
    """
    loss_type = (loss_type or 'supcon').lower()
    if loss_type in ('both', 'supcon+rnc', 'rnc+supcon'):
        return {
            'supcon': SupConLoss(temperature=supcon_temperature),
            'rnc': RNCLoss(temperature=rnc_temperature, similarity=rnc_similarity),
            'deepsad': None,
        }
    if loss_type == 'supcon':
        return {
            'supcon': SupConLoss(temperature=supcon_temperature),
            'rnc': None,
            'deepsad': None,
        }
    if loss_type == 'rnc':
        return {
            'supcon': None,
            'rnc': RNCLoss(temperature=rnc_temperature, similarity=rnc_similarity),
            'deepsad': None,
        }
    if loss_type == 'deepsad':
        return {
            'supcon': None,
            'rnc': None,
            'deepsad': DeepSADLoss(margin=deepsad_margin, repulsion_weight=deepsad_repulsion_weight),
        }
    if loss_type == 'none':
        return {'supcon': None, 'rnc': None, 'deepsad': None}
    raise ValueError(
        f"Unknown contrastive_loss {loss_type!r}. "
        "Use 'supcon', 'rnc', 'both', 'deepsad', or 'none'."
    )
