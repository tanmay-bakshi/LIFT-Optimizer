"""
lift_optimizer module
=====================

This module provides an implementation of the **Low‑rank Informed Sparse
Fine‑Tuning** (LIFT) algorithm described in the paper *“LIFT the Veil for
the Truth: Principal Weights Emerge after Rank Reduction for
Reasoning‑Focused Supervised Fine‑Tuning”*【792723377939635†L1743-L1785】.  The LIFT method
identifies **principal weights** within a weight matrix by first applying
a low‑rank approximation (via singular value decomposition) and then
selecting those elements with the highest magnitude in the
approximation【792723377939635†L1743-L1785】.  During fine‑tuning only these selected
parameters are updated; all other weights remain frozen.  The set of
selected indices (the *mask*) is updated periodically throughout
training to account for shifts in the principal weights as learning
progresses【792723377939635†L1803-L1819】.

This implementation follows the pseudocode presented in Appendix A of
the paper【792723377939635†L1743-L1785】.  It provides a custom PyTorch optimizer
``LIFTSparseAdamW`` that wraps the AdamW update while applying LIFT’s
sparse update rule.  The optimizer computes a low‑rank approximation
for each eligible parameter, constructs a binary mask selecting the
largest entries of that approximation, and then maintains first and
second moments only for the selected elements.  At each update step
the mask can be refreshed at a user‑defined interval.  Weight decay
and bias correction are handled following the standard AdamW
procedure.

Key algorithmic components
-------------------------

* **Low‑rank approximation:** For a parameter matrix :math:`W ∈ ℝ^{m×n}`,
  a truncated singular value decomposition (SVD) of rank ``filter_rank``
  is performed.  Only the top singular vectors and values are retained,
  giving :math:`W' = U_r Σ_r V_r^⊤`【792723377939635†L1743-L1785】.  This step filters out
  high‑order noise and yields a smoother estimate of the weight matrix.

* **Principal weight selection:** The magnitude of each element in
  :math:`W'` is taken (optionally the signed values may be used),
  flattened into a vector, and the top ``k`` values are selected.
  These indices form a binary mask identifying the *principal
  weights*【792723377939635†L1743-L1785】.  The value of ``k`` can be determined
  either by specifying a fixed sparsity ratio or by setting a
  ``num_principal`` parameter that scales with the matrix dimensions.

* **Sparse optimization:** During each optimization step the gradient
  ``g`` is restricted to the selected indices.  First and second
  moment vectors ``m`` and ``v`` (as in Adam) are maintained only for
  these entries, significantly reducing memory usage【792723377939635†L1743-L1785】.  Bias
  correction is then applied and the weight update is executed on the
  selected elements.  Non‑selected elements are left unchanged.

* **Dynamic mask update:** The principal weights may change as the
  model learns.  Therefore the mask is recomputed every
  ``update_interval`` steps.  On update the existing first and second
  moment vectors are expanded back to full tensors, then the new mask
  extracts updated moments corresponding to the new principal
  weights【792723377939635†L1743-L1785】.  An appropriate interval should balance
  stability and adaptability; very small intervals may undertrain
  weights while very large intervals may miss newly emerging principal
  weights【792723377939635†L1803-L1819】.

The class defined in this file can be used with any PyTorch model.  It
accepts optional callbacks to decide which parameters should be
subjected to LIFT and which should be fully fine‑tuned.  By default
all two‑dimensional parameters (e.g. linear and convolutional weight
matrices) are sparsified while biases and one‑dimensional parameters
are updated densely.

The API is kept simple: users instantiate the optimizer with a model’s
parameters and supply LIFT‑specific hyperparameters such as
``filter_rank``, ``sparsity_ratio`` or ``num_principal``, and
``update_interval``.  Comprehensive docstrings accompany each method
to guide proper use.

"""

from __future__ import annotations

import math
from typing import Iterable, Callable, Optional, Dict, Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class LIFTSparseAdamW(Optimizer):
    """AdamW optimizer with Low‑rank Informed Sparse Fine‑Tuning (LIFT).

    This optimizer implements the LIFT algorithm by sparsifying updates
    to only a subset of entries within eligible parameter matrices.  It
    is a drop‑in replacement for :class:`torch.optim.AdamW` when
    fine‑tuning large models with memory constraints.  The selected
    entries (the *mask*) are obtained from a low‑rank approximation of
    each weight matrix followed by magnitude‑based top‑`k` selection.

    Parameters are divided into two classes:

    * **Sparse parameters:** Parameters for which the update mask is
      computed and only masked entries are updated.  These are
      typically two‑dimensional weight matrices.  The decision of
      whether a parameter is sparse can be customized via
      ``param_filter``.
    * **Dense parameters:** Parameters that are always fully updated,
      such as biases or one‑dimensional tensors.

    Hyperparameters controlling LIFT include:

    :param params: An iterable of parameters to optimize.  This is
        identical to the ``params`` argument of standard PyTorch
        optimizers.  Internally, parameters are grouped into sparse and
        dense sets depending on ``param_filter``.
    :param float lr: Learning rate.  Defaults to ``1e-4``.
    :param tuple betas: Adam momentum coefficients ``(β1, β2)``.
    :param float eps: Numerical stability constant ``ϵ`` added to the
        denominator in the Adam update.
    :param float weight_decay: Optional weight decay applied to
        selected entries.  Weight decay is decoupled as in AdamW.
    :param int filter_rank: Rank ``r`` of the truncated SVD used to
        compute low‑rank approximations.  A larger rank retains more
        information from the original weight matrix but increases the
        cost of SVD.  Reasonable values are between 4 and 32; the
        authors report using ``r=8``【792723377939635†L1743-L1785】.
    :param Optional[float] sparsity_ratio: Fraction of entries to
        update in each sparse parameter.  If provided, the top
        ``sparsity_ratio * n`` entries (where ``n`` is the number of
        elements in the matrix) are selected.  Exactly one of
        ``sparsity_ratio`` or ``num_principal`` must be specified.
    :param Optional[int] num_principal: The number of principal weights
        to update per parameter, computed as ``(m + n) * k`` where
        ``m`` and ``n`` are the row and column dimensions of the
        parameter and ``k`` is an integer hyperparameter.  This mode
        mirrors the paper’s use of the LoRA rank (called
        ``lora_rank``) to determine the number of selected entries
       【792723377939635†L1743-L1785】.  Exactly one of ``num_principal`` or
        ``sparsity_ratio`` must be specified.
    :param int update_interval: Frequency (in steps) at which the
        principal mask is recomputed.  Dynamic mask updates help the
        algorithm adapt to shifting principal weights during fine‑tuning.
        Intervals that are too small may cause under‑training of
        weights, whereas intervals that are too large may miss newly
        emerging important weights【792723377939635†L1803-L1819】.  Empirically
        100–500 steps are effective.
    :param bool use_abs: If ``True``, principal weights are selected
        based on absolute magnitude.  If ``False``, signed values are
        used.  Using absolute values matches the paper’s description.
    :param Optional[Callable[[str, Tensor], bool]] param_filter: A
        callable that takes a parameter name and tensor and returns
        ``True`` if the parameter should be sparsified.  By default
        parameters with ``dim() >= 2`` are sparsified.
    :param Optional[torch.device] device: Device on which SVD
        computations will be performed.  If ``None``, defaults to the
        current device of the parameter tensor.  It is recommended to
        use the same device as your model (e.g., GPU) for speed.

    Raises:
        ValueError: If neither or both of ``sparsity_ratio`` and
        ``num_principal`` are provided.

    Example usage::

        model = MyLargeModel()
        optimizer = LIFTSparseAdamW(
            model.named_parameters(),
            lr=2e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            filter_rank=8,
            num_principal=4,  # selects (m + n) * 4 principal weights
            update_interval=200,
        )
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    The optimizer transparently handles both sparse and dense
    parameters.  When using this optimizer you should not wrap your
    model with additional adapter modules; LIFT is applied directly on
    the existing weights.

    """

    def __init__(
        self,
        params: Iterable[tuple[str, Tensor]] | Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        filter_rank: int = 8,
        sparsity_ratio: Optional[float] = None,
        num_principal: Optional[int] = None,
        update_interval: int = 200,
        use_abs: bool = True,
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # Validate hyperparameters
        if sparsity_ratio is None and num_principal is None:
            raise ValueError("Either sparsity_ratio or num_principal must be provided.")
        if sparsity_ratio is not None and num_principal is not None:
            raise ValueError("Specify only one of sparsity_ratio or num_principal, not both.")
        if not 0.0 < lr:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Beta1 must be in [0, 1).")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Beta2 must be in [0, 1).")
        if filter_rank <= 0:
            raise ValueError("filter_rank must be a positive integer.")
        if sparsity_ratio is not None and not (0.0 < sparsity_ratio <= 1.0):
            raise ValueError("sparsity_ratio must be in (0, 1].")
        if num_principal is not None and num_principal <= 0:
            raise ValueError("num_principal must be a positive integer.")
        if update_interval <= 0:
            raise ValueError("update_interval must be a positive integer.")

        # Prepare parameter groups; we allow either an iterable of parameters or
        # an iterable of (name, parameter) tuples for convenience.  We need
        # names to allow filtering but accept unnamed as well.
        named_parameters: list[tuple[str, Tensor]] = []
        if isinstance(params, Iterable):
            # The iterable may be of parameters or (name, param) pairs.  If
            # elements are not tuples we synthesise names using index.
            index = 0
            for item in params:
                if isinstance(item, tuple) and len(item) == 2:
                    name, tensor = item
                    named_parameters.append((name, tensor))
                else:
                    named_parameters.append((f"param_{index}", item))
                    index += 1

        # Determine default param_filter if not supplied.  Only tensors with
        # dimensionality ≥2 are sparsified by default.
        def default_filter(name: str, tensor: Tensor) -> bool:
            return tensor.dim() >= 2

        self.param_filter = param_filter or default_filter

        self.use_abs = use_abs
        self.filter_rank = filter_rank
        self.sparsity_ratio = sparsity_ratio
        self.num_principal = num_principal
        self.update_interval = update_interval
        self.device = device

        # Create separate lists for dense and sparse parameters.  Dense
        # parameters will be updated using the standard AdamW update.
        dense_params: list[Tensor] = []
        sparse_params: list[Tensor] = []
        param_names: Dict[Tensor, str] = {}
        for name, p in named_parameters:
            param_names[p] = name
            if self.param_filter(name, p):
                sparse_params.append(p)
            else:
                dense_params.append(p)

        # Build param_groups for the underlying Optimizer
        param_groups = []
        if sparse_params:
            param_groups.append({"params": sparse_params, "sparse": True})
        if dense_params:
            param_groups.append({"params": dense_params, "sparse": False})

        defaults: Dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(param_groups, defaults)

        # Initialize per‑parameter state for sparse parameters
        self.state: Dict[Tensor, Dict[str, Any]] = {}  # type: ignore[assignment]
        self._step: int = 0
        for group in self.param_groups:
            if not group.get("sparse", False):
                continue
            for p in group["params"]:
                # allocate state dictionaries
                self.state[p] = {}
                # compute initial mask and allocate m_update/v_update
                mask = self._compute_mask(p)
                # flatten mask indices for convenience
                self.state[p]["mask"] = mask
                num_selected = mask.sum().item()
                # first and second moment vectors for selected entries
                self.state[p]["m_update"] = torch.zeros(num_selected, dtype=p.dtype, device=p.device)
                self.state[p]["v_update"] = torch.zeros(num_selected, dtype=p.dtype, device=p.device)
                # store old selected indices for when mask is updated
                self.state[p]["prev_mask"] = mask
                self.state[p]["step"] = 0

    def _compute_mask(self, param: Tensor) -> Tensor:
        """Compute a binary mask of principal weights for a parameter.

        A truncated singular value decomposition is performed on
        ``param``.  The top ``filter_rank`` singular values and
        corresponding singular vectors are used to reconstruct an
        approximate weight matrix.  The entries with the largest
        magnitude (as determined by either ``sparsity_ratio`` or
        ``num_principal``) are selected to form the mask.

        :param param: Weight matrix whose mask should be computed.
        :returns: A boolean tensor with the same shape as ``param``
            indicating selected entries.
        :except RuntimeError: If SVD fails to converge.
        """
        # Move the tensor to the desired device for SVD computation
        p = param.detach()
        device = self.device or p.device
        # convert to float32 for stable SVD (common practice)
        data = p.to(device=device, dtype=torch.float32)
        # Perform truncated SVD
        # We use torch.linalg.svd rather than svd_lowrank to guarantee
        # precision.  For very large matrices one could consider
        # torch.linalg.svdvals or randomized algorithms.
        U, S, Vh = torch.linalg.svd(data, full_matrices=False)
        r = min(self.filter_rank, S.shape[0])
        U_r: Tensor = U[:, :r]
        S_r: Tensor = S[:r]
        V_r: Tensor = Vh[:r, :]
        # reconstruct rank‑r approximation
        # Equivalent to U_r @ diag(S_r) @ V_r
        # We'll multiply S_r across columns of V_r for efficiency
        reconstructed: Tensor = (U_r * S_r.unsqueeze(0)) @ V_r
        # Determine how many entries to select
        total_elems = reconstructed.numel()
        if self.sparsity_ratio is not None:
            k = max(1, int(total_elems * self.sparsity_ratio))
        else:
            # number of principal weights: (m + n) * num_principal
            m, n = reconstructed.shape
            k = min(total_elems, (m + n) * self.num_principal)  # type: ignore[operator]
            k = max(1, int(k))
        # Flatten and select top‐k indices based on magnitude
        flat: Tensor
        if self.use_abs:
            flat = reconstructed.abs().reshape(-1)
        else:
            flat = reconstructed.reshape(-1)
        # Avoid extremely small k relative to matrix size
        topk_vals = torch.topk(flat, k=k, largest=True)
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[topk_vals.indices] = True
        return mask_flat.view_as(param)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.

        This method updates both sparse and dense parameters.  For sparse
        parameters the update is restricted to entries selected by the
        current mask; first and second moment vectors are maintained
        only for these entries.  The mask is recomputed every
        ``update_interval`` steps.  Dense parameters are updated
        according to the standard AdamW rule.

        :param closure: Optional closure re‑evaluating the model and
            returning the loss.  Not used in this implementation.
        :returns: The loss returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # Increment global step
        self._step += 1

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            is_sparse: bool = group.get("sparse", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad: Tensor = p.grad
                if not is_sparse:
                    # Standard dense AdamW update
                    state = self.state.setdefault(p, {})
                    if not state:
                        # Initialize dense moments
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["step"] += 1
                    exp_avg: Tensor = state["exp_avg"]
                    exp_avg_sq: Tensor = state["exp_avg_sq"]
                    # Decay the first and second moment running average coefficients
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    # Compute bias‑corrected update
                    denom: Tensor = exp_avg_sq.sqrt().add_(eps)
                    update: Tensor = exp_avg / denom
                    # Apply weight decay
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)
                    # Update parameter
                    p.data.add_(update, alpha=-step_size)
                else:
                    # Sparse update using LIFT
                    state = self.state[p]
                    state["step"] += 1
                    # Update mask at specified interval
                    if (self._step % self.update_interval) == 0:
                        # Expand previous moments into full tensor
                        old_mask: Tensor = state["mask"]
                        m_full = torch.zeros_like(p.data)
                        v_full = torch.zeros_like(p.data)
                        # scatter existing moments back to full tensors
                        m_full[old_mask] = state["m_update"]
                        v_full[old_mask] = state["v_update"]
                        # compute new mask
                        new_mask = self._compute_mask(p)
                        # extract updated moments for new mask
                        new_m_update = m_full[new_mask]
                        new_v_update = v_full[new_mask]
                        # assign new mask and moment vectors
                        state["mask"] = new_mask
                        state["m_update"] = new_m_update
                        state["v_update"] = new_v_update
                        state["prev_mask"] = new_mask
                    # Restrict gradient to selected indices
                    mask: Tensor = state["mask"]
                    grad_selected: Tensor = grad[mask]
                    # Retrieve and update first and second moments for selected entries
                    m_update: Tensor = state["m_update"]
                    v_update: Tensor = state["v_update"]
                    # Update biased first moment estimate
                    m_update.mul_(beta1).add_(grad_selected, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    v_update.mul_(beta2).addcmul_(grad_selected, grad_selected, value=1 - beta2)
                    # Compute bias‑corrected first and second moment estimates
                    step_i: int = state["step"]
                    bias_correction1 = 1 - beta1 ** step_i
                    bias_correction2 = 1 - beta2 ** step_i
                    m_hat: Tensor = m_update / bias_correction1
                    v_hat: Tensor = v_update / bias_correction2
                    denom: Tensor = v_hat.sqrt().add_(eps)
                    update_vals: Tensor = m_hat / denom
                    # Apply weight decay to selected indices
                    if weight_decay != 0:
                        p.data[mask].mul_(1 - lr * weight_decay)
                    # Apply update
                    p.data[mask] = p.data[mask] - lr * update_vals
                    # Save updated moments back to state
                    state["m_update"] = m_update
                    state["v_update"] = v_update
        return loss
