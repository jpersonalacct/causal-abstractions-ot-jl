"""Shared pair-bank data structures for hierarchical equality counterfactual splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_TARGET_VARS
from .scm import (
    EqualityProblem,
    compute_counterfactual_labels,
    compute_states_for_rows,
    rows_to_inputs_embeds,
    verify_counterfactual_labels_with_scm,
)

PAIR_POLICY_VARS = ("WX", "YZ")
DEFAULT_PAIR_POLICY_TARGET = "any"


@dataclass(frozen=True)
class PairBank:
    """Shared base/source pair split with factual and counterfactual labels."""

    split: str
    seed: int
    base_rows: torch.Tensor
    source_rows: torch.Tensor
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
        return int(self.base_inputs.shape[0])

    def metadata(self) -> dict[str, Any]:
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
        if variable_name not in bank.cf_labels_by_var:
            raise KeyError(f"Unknown variable {variable_name}")
        self.bank = bank
        self.variable_name = variable_name
        self.variable_index = DEFAULT_TARGET_VARS.index(variable_name)

    def __len__(self) -> int:
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.bank.base_inputs[index],
            "source_input_ids": self.bank.source_inputs[index],
            "labels": self.bank.cf_labels_by_var[self.variable_name][index],
            "base_labels": self.bank.base_labels[index],
            "intervention_id": torch.tensor(self.variable_index, dtype=torch.long),
        }


def _minimum_pool_size(size: int) -> int:
    if size <= 0:
        return 2
    pool_size = 2
    while pool_size * (pool_size - 1) < size:
        pool_size += 1
    return pool_size


def _sample_unique_entity_rows(size: int, seed: int, num_entities: int) -> np.ndarray:
    total = num_entities ** 4
    if size > total:
        raise ValueError(
            f"Requested pool_size={size}, but only {total} unique four-entity rows exist"
        )
    rng = np.random.default_rng(seed)
    values = rng.choice(total, size=size, replace=False)
    rows = np.zeros((size, 4), dtype=np.int64)
    for column in range(3, -1, -1):
        rows[:, column] = values % num_entities
        values = values // num_entities
    return rows


def _compute_changed_flags(
    base_labels: np.ndarray,
    cf_labels_by_var: dict[str, np.ndarray],
    policy_vars: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
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
    changed_wx = changed_by_var.get("WX")
    changed_yz = changed_by_var.get("YZ")
    if changed_wx is None or changed_yz is None:
        raise ValueError("Pair policy targets require equality policy vars ('WX', 'YZ')")

    if pair_policy_target == "any":
        return np.logical_or(changed_wx, changed_yz)
    if pair_policy_target == "WX":
        return changed_wx
    if pair_policy_target == "YZ":
        return changed_yz
    if pair_policy_target == "both":
        return np.logical_and(changed_wx, changed_yz)
    if pair_policy_target == "WX_only":
        return np.logical_and(changed_wx, np.logical_not(changed_yz))
    if pair_policy_target == "YZ_only":
        return np.logical_and(changed_yz, np.logical_not(changed_wx))
    raise ValueError(f"Unsupported pair_policy_target={pair_policy_target}")


def _print_pair_bank_stats(split: str, stats: dict[str, Any]) -> None:
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
    selected: list[int] = []
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
        f"Could not construct {size} mixed pairs; increase pair_pool_size or relax the policy"
    )


def build_pair_bank_from_rows(
    problem: EqualityProblem,
    *,
    base_rows: np.ndarray,
    source_rows: np.ndarray,
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
    base_rows = np.asarray(base_rows, dtype=np.int64)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if base_rows.shape != source_rows.shape:
        raise ValueError(
            f"Expected matching base/source shapes, got {base_rows.shape} and {source_rows.shape}"
        )

    base_states = compute_states_for_rows(base_rows)
    source_states = compute_states_for_rows(source_rows)
    cf_labels_np = compute_counterfactual_labels(base_states, source_states)
    changed_by_var_np, changed_any_np = _compute_changed_flags(
        base_labels=base_states["O"],
        cf_labels_by_var=cf_labels_np,
        policy_vars=pair_policy_vars,
    )
    stats = _summarize_pair_changes(changed_by_var_np, changed_any_np, pair_policy_vars)

    if verify_with_scm:
        verify_counterfactual_labels_with_scm(problem, base_rows, source_rows, cf_labels_np)

    bank = PairBank(
        split=split,
        seed=seed,
        base_rows=torch.tensor(base_rows, dtype=torch.long),
        source_rows=torch.tensor(source_rows, dtype=torch.long),
        base_inputs=rows_to_inputs_embeds(base_rows, problem.input_var_order, problem.entity_vectors),
        source_inputs=rows_to_inputs_embeds(source_rows, problem.input_var_order, problem.entity_vectors),
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
    problem: EqualityProblem,
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
    rng = np.random.default_rng(seed)
    required_pool_size = _minimum_pool_size(size)
    resolved_pool_size = max(
        required_pool_size,
        required_pool_size if pair_pool_size is None else int(pair_pool_size),
    )
    row_pool = _sample_unique_entity_rows(
        resolved_pool_size,
        seed + 17,
        num_entities=problem.num_entities,
    )

    ordered_pairs = [
        (base_index, source_index)
        for base_index in range(resolved_pool_size)
        for source_index in range(resolved_pool_size)
        if base_index != source_index
    ]
    pair_order = rng.permutation(len(ordered_pairs))
    base_indices = np.asarray([ordered_pairs[index][0] for index in pair_order], dtype=np.int64)
    source_indices = np.asarray([ordered_pairs[index][1] for index in pair_order], dtype=np.int64)

    candidate_base_rows = row_pool[base_indices]
    candidate_source_rows = row_pool[source_indices]
    candidate_base_states = compute_states_for_rows(candidate_base_rows)
    candidate_source_states = compute_states_for_rows(candidate_source_rows)
    candidate_cf_labels_np = compute_counterfactual_labels(candidate_base_states, candidate_source_states)
    changed_by_var_np, _ = _compute_changed_flags(
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

    return build_pair_bank_from_rows(
        problem,
        base_rows=candidate_base_rows[selected_indices],
        source_rows=candidate_source_rows[selected_indices],
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
