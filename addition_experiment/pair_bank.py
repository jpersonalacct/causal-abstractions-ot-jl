"""Shared pair-bank data structures for base/source counterfactual splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_TARGET_VARS
from .scm import (
    AdditionProblem,
    compute_counterfactual_labels,
    compute_states_for_digits,
    digits_to_inputs_embeds,
    verify_counterfactual_labels_with_scm,
)

PAIR_POLICY_VARS = ("C1", "C2")
DEFAULT_PAIR_POLICY_TARGET = "any"


@dataclass(frozen=True)
class PairBank:
    """Shared base/source pair split with factual and counterfactual labels."""

    split: str
    seed: int
    base_digits: torch.Tensor
    source_digits: torch.Tensor
    base_inputs: torch.Tensor
    source_inputs: torch.Tensor
    base_labels: torch.Tensor
    cf_labels_by_var: dict[str, torch.Tensor]
    changed_by_var: dict[str, torch.Tensor]
    changed_any: torch.Tensor
    pair_policy: str
    pair_policy_target: str
    mixed_positive_fraction: float
    target_vars: tuple[str, ...]
    pair_policy_vars: tuple[str, ...]
    pair_pool_size: int | None
    pair_stats: dict[str, Any]

    @property
    def size(self) -> int:
        """Return the number of base/source pairs in the bank."""
        return int(self.base_inputs.shape[0])

    def metadata(self) -> dict[str, Any]:
        """Return a compact summary of the pair-bank split."""
        return {
            "split": self.split,
            "seed": self.seed,
            "size": self.size,
            "pair_policy": self.pair_policy,
            "pair_policy_target": self.pair_policy_target,
            "mixed_positive_fraction": self.mixed_positive_fraction,
            "target_vars": list(self.target_vars),
            "pair_policy_vars": list(self.pair_policy_vars),
            "pair_pool_size": self.pair_pool_size,
            "pair_stats": self.pair_stats,
        }


class PairBankVariableDataset(Dataset):
    """Dataset view exposing one abstract variable's counterfactual labels."""

    def __init__(self, bank: PairBank, variable_name: str):
        """Wrap one abstract variable view of a shared pair bank for DAS."""
        if variable_name not in bank.cf_labels_by_var:
            raise KeyError(f"Unknown variable {variable_name}")
        self.bank = bank
        self.variable_name = variable_name
        self.variable_index = DEFAULT_TARGET_VARS.index(variable_name)

    def __len__(self) -> int:
        """Return the number of examples exposed by this dataset view."""
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch one base/source intervention example for the chosen variable."""
        return {
            "input_ids": self.bank.base_inputs[index],
            "source_input_ids": self.bank.source_inputs[index],
            "labels": self.bank.cf_labels_by_var[self.variable_name][index],
            "base_labels": self.bank.base_labels[index],
            "intervention_id": torch.tensor(self.variable_index, dtype=torch.long),
        }


def _minimum_pool_size(size: int) -> int:
    """Return the smallest pool that can realize `size` ordered distinct pairs."""
    if size <= 0:
        return 2
    pool_size = 2
    while pool_size * (pool_size - 1) < size:
        pool_size += 1
    return pool_size


def _sample_unique_digit_rows(size: int, seed: int) -> np.ndarray:
    """Sample unique four-digit rows without replacement from the 10k support."""
    if size > 10_000:
        raise ValueError(f"Requested pool_size={size}, but only 10000 unique digit rows exist")
    rng = np.random.default_rng(seed)
    values = rng.choice(10_000, size=size, replace=False)
    digits = np.zeros((size, 4), dtype=np.int64)
    for column in range(3, -1, -1):
        digits[:, column] = values % 10
        values = values // 10
    return digits


def _compute_changed_flags(
    base_labels: np.ndarray,
    cf_labels_by_var: dict[str, np.ndarray],
    policy_vars: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mark whether each policy-variable interchange changes the final label."""
    changed_by_var = {
        variable: cf_labels_by_var[variable] != base_labels for variable in policy_vars
    }
    if not policy_vars:
        changed_any = np.zeros_like(base_labels, dtype=bool)
    else:
        changed_any = np.logical_or.reduce([changed_by_var[variable] for variable in policy_vars])
    return changed_by_var, changed_any


def _summarize_pair_changes(
    changed_by_var: dict[str, np.ndarray],
    changed_any: np.ndarray,
    policy_vars: tuple[str, ...],
) -> dict[str, Any]:
    """Build split-level pair statistics for logging and JSON output."""
    total_pairs = int(changed_any.shape[0])
    per_variable = {}
    for variable in policy_vars:
        changed_count = int(changed_by_var[variable].sum())
        unchanged_count = total_pairs - changed_count
        per_variable[variable] = {
            "changed_count": changed_count,
            "unchanged_count": unchanged_count,
            "changed_rate": float(changed_count / total_pairs) if total_pairs else 0.0,
        }
    changed_any_count = int(changed_any.sum())
    return {
        "total_pairs": total_pairs,
        "changed_any_count": changed_any_count,
        "unchanged_any_count": total_pairs - changed_any_count,
        "changed_any_rate": float(changed_any_count / total_pairs) if total_pairs else 0.0,
        "per_variable": per_variable,
    }


def _compute_policy_positive_mask(
    changed_by_var: dict[str, np.ndarray],
    pair_policy_target: str,
) -> np.ndarray:
    """Resolve the boolean mask treated as positive by the pair policy."""
    changed_c1 = changed_by_var.get("C1")
    changed_c2 = changed_by_var.get("C2")
    if changed_c1 is None or changed_c2 is None:
        raise ValueError("Pair policy targets require carry policy vars ('C1', 'C2')")

    if pair_policy_target == "any":
        return np.logical_or(changed_c1, changed_c2)
    if pair_policy_target == "C1":
        return changed_c1
    if pair_policy_target == "C2":
        return changed_c2
    if pair_policy_target == "both":
        return np.logical_and(changed_c1, changed_c2)
    if pair_policy_target == "C1_only":
        return np.logical_and(changed_c1, np.logical_not(changed_c2))
    if pair_policy_target == "C2_only":
        return np.logical_and(changed_c2, np.logical_not(changed_c1))
    raise ValueError(f"Unsupported pair_policy_target={pair_policy_target}")


def _print_pair_bank_stats(split: str, stats: dict[str, Any]) -> None:
    """Print one compact summary per split plus per-variable change rates."""
    print(
        f"{split} pair bank "
        f"| total_pairs={int(stats['total_pairs'])} "
        f"| changed_any={int(stats['changed_any_count'])} "
        f"| unchanged_any={int(stats['unchanged_any_count'])}"
    )
    for variable, variable_stats in stats["per_variable"].items():
        print(
            f"{split} pair bank [{variable}] "
            f"| changed={int(variable_stats['changed_count'])} "
            f"| unchanged={int(variable_stats['unchanged_count'])} "
            f"| changed_rate={float(variable_stats['changed_rate']):.4f}"
        )


def _select_pair_indices(
    *,
    positive_mask: np.ndarray,
    size: int,
    pair_policy: str,
    mixed_positive_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose a final ordered-pair subset under the requested policy."""
    if size < 0:
        raise ValueError(f"Expected non-negative size, got {size}")
    num_candidates = int(positive_mask.shape[0])
    if size > num_candidates:
        raise ValueError(
            f"Requested size={size}, but only {num_candidates} candidate ordered pairs are available"
        )

    shuffled = rng.permutation(num_candidates)
    if pair_policy == "unfiltered":
        return shuffled[:size]

    selected: list[int] = []
    if pair_policy != "mixed":
        raise ValueError(f"Unsupported pair_policy={pair_policy}")

    if not 0.0 <= float(mixed_positive_fraction) <= 1.0:
        raise ValueError(
            "Expected mixed_positive_fraction to lie in [0, 1], "
            f"got {mixed_positive_fraction}"
        )
    positive_target = int(np.floor(size * float(mixed_positive_fraction) + 0.5))
    positive_target = min(max(positive_target, 0), size)
    negative_target = size - positive_target
    positive_count = 0
    negative_count = 0
    for index in shuffled:
        is_positive = bool(positive_mask[index])
        if is_positive:
            if positive_count >= positive_target:
                continue
            positive_count += 1
        else:
            if negative_count >= negative_target:
                continue
            negative_count += 1
        selected.append(int(index))
        if len(selected) == size:
            return np.asarray(selected, dtype=np.int64)

    raise ValueError(
        f"Could not construct {size} mixed pairs; "
        "increase pair_pool_size or relax the policy"
    )


def build_pair_bank_from_digits(
    problem: AdditionProblem,
    *,
    base_digits: np.ndarray,
    source_digits: np.ndarray,
    seed: int,
    split: str,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    pair_policy_vars: tuple[str, ...] = PAIR_POLICY_VARS,
    pair_policy: str = "unfiltered",
    pair_policy_target: str = DEFAULT_PAIR_POLICY_TARGET,
    mixed_positive_fraction: float = 0.5,
    pair_pool_size: int | None = None,
    verify_with_scm: bool = False,
) -> PairBank:
    """Create a pair bank from explicit ordered base/source digit rows."""
    base_digits = np.asarray(base_digits, dtype=np.int64)
    source_digits = np.asarray(source_digits, dtype=np.int64)
    if base_digits.shape != source_digits.shape:
        raise ValueError(
            f"Expected matching base/source shapes, got {base_digits.shape} and {source_digits.shape}"
        )

    base_states = compute_states_for_digits(base_digits)
    source_states = compute_states_for_digits(source_digits)
    cf_labels_np = compute_counterfactual_labels(base_states, source_states)
    changed_by_var_np, changed_any_np = _compute_changed_flags(
        base_labels=base_states["O"],
        cf_labels_by_var=cf_labels_np,
        policy_vars=pair_policy_vars,
    )
    stats = _summarize_pair_changes(changed_by_var_np, changed_any_np, pair_policy_vars)

    if verify_with_scm:
        verify_counterfactual_labels_with_scm(problem, base_digits, source_digits, cf_labels_np)

    bank = PairBank(
        split=split,
        seed=seed,
        base_digits=torch.tensor(base_digits, dtype=torch.long),
        source_digits=torch.tensor(source_digits, dtype=torch.long),
        base_inputs=digits_to_inputs_embeds(base_digits, problem.input_var_order),
        source_inputs=digits_to_inputs_embeds(source_digits, problem.input_var_order),
        base_labels=torch.tensor(base_states["O"], dtype=torch.long),
        cf_labels_by_var={
            var: torch.tensor(cf_labels_np[var], dtype=torch.long) for var in DEFAULT_TARGET_VARS
        },
        changed_by_var={
            variable: torch.tensor(changed_by_var_np[variable], dtype=torch.bool)
            for variable in pair_policy_vars
        },
        changed_any=torch.tensor(changed_any_np, dtype=torch.bool),
        pair_policy=pair_policy,
        pair_policy_target=pair_policy_target,
        mixed_positive_fraction=float(mixed_positive_fraction),
        target_vars=tuple(target_vars),
        pair_policy_vars=tuple(pair_policy_vars),
        pair_pool_size=pair_pool_size,
        pair_stats=stats,
    )
    _print_pair_bank_stats(split, stats)
    return bank


def build_pair_bank(
    problem: AdditionProblem,
    size: int,
    seed: int,
    split: str,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    pair_policy_vars: tuple[str, ...] = PAIR_POLICY_VARS,
    pair_policy: str = "unfiltered",
    pair_policy_target: str = DEFAULT_PAIR_POLICY_TARGET,
    mixed_positive_fraction: float = 0.5,
    pair_pool_size: int | None = None,
    verify_with_scm: bool = False,
) -> PairBank:
    """Create a deterministic base/source pair bank and counterfactual labels."""
    rng = np.random.default_rng(seed)
    required_pool_size = _minimum_pool_size(size)
    resolved_pool_size = max(
        required_pool_size,
        required_pool_size if pair_pool_size is None else int(pair_pool_size),
    )
    digit_pool = _sample_unique_digit_rows(resolved_pool_size, seed + 17)

    ordered_pairs = [(base_index, source_index) for base_index in range(resolved_pool_size) for source_index in range(resolved_pool_size) if base_index != source_index]
    pair_order = rng.permutation(len(ordered_pairs))
    base_indices = np.asarray([ordered_pairs[index][0] for index in pair_order], dtype=np.int64)
    source_indices = np.asarray([ordered_pairs[index][1] for index in pair_order], dtype=np.int64)

    candidate_base_digits = digit_pool[base_indices]
    candidate_source_digits = digit_pool[source_indices]
    candidate_base_states = compute_states_for_digits(candidate_base_digits)
    candidate_source_states = compute_states_for_digits(candidate_source_digits)
    candidate_cf_labels_np = compute_counterfactual_labels(candidate_base_states, candidate_source_states)
    changed_by_var_np, changed_any_np = _compute_changed_flags(
        base_labels=candidate_base_states["O"],
        cf_labels_by_var=candidate_cf_labels_np,
        policy_vars=pair_policy_vars,
    )
    positive_mask = _compute_policy_positive_mask(changed_by_var_np, pair_policy_target)
    selected_indices = _select_pair_indices(
        positive_mask=positive_mask,
        size=size,
        pair_policy=pair_policy,
        mixed_positive_fraction=mixed_positive_fraction,
        rng=rng,
    )

    return build_pair_bank_from_digits(
        problem,
        base_digits=candidate_base_digits[selected_indices],
        source_digits=candidate_source_digits[selected_indices],
        seed=seed,
        split=split,
        target_vars=target_vars,
        pair_policy_vars=pair_policy_vars,
        pair_policy=pair_policy,
        pair_policy_target=pair_policy_target,
        mixed_positive_fraction=mixed_positive_fraction,
        pair_pool_size=resolved_pool_size,
        verify_with_scm=verify_with_scm,
    )
