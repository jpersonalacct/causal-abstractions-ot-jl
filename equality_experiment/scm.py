"""Symbolic hierarchical equality SCM and tensor conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import _env  # noqa: F401

from pyvene import CausalModel

from .constants import (
    CANONICAL_INPUT_VARS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_NUM_ENTITIES,
    DEFAULT_TARGET_VARS,
)


@dataclass(frozen=True)
class EqualityProblem:
    """Bundle of the symbolic SCM and the entity embedding table."""

    causal_model: CausalModel
    input_var_order: tuple[str, ...]
    entity_vectors: np.ndarray
    embedding_dim: int
    num_entities: int


def build_entity_vectors(
    num_entities: int = DEFAULT_NUM_ENTITIES,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    seed: int = 0,
) -> np.ndarray:
    """Create the fixed random entity vectors used as low-level inputs."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(num_entities, embedding_dim)).astype(np.float32)


def as_entity_index(value: Any, entity_vectors: np.ndarray) -> int:
    """Decode an entity index from either a scalar id or one entity vector."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return int(arr[0])
    distances = np.linalg.norm(entity_vectors - arr.reshape(1, -1), axis=1)
    return int(distances.argmin())


def build_equality_causal_model(entity_vectors: np.ndarray) -> CausalModel:
    """Construct the hierarchical equality SCM."""
    variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]
    values = {variable: [vector.copy() for vector in entity_vectors] for variable in CANONICAL_INPUT_VARS}
    values["WX"] = [0, 1]
    values["YZ"] = [0, 1]
    values["O"] = [0, 1]
    parents = {
        "W": [],
        "X": [],
        "Y": [],
        "Z": [],
        "WX": ["W", "X"],
        "YZ": ["Y", "Z"],
        "O": ["WX", "YZ"],
    }

    def filler() -> np.ndarray:
        return entity_vectors[0]

    functions = {
        "W": filler,
        "X": filler,
        "Y": filler,
        "Z": filler,
        "WX": lambda w, x: int(np.array_equal(w, x)),
        "YZ": lambda y, z: int(np.array_equal(y, z)),
        "O": lambda wx, yz: int(int(wx) == int(yz)),
    }
    return CausalModel(variables, values, parents, functions)


def assignment_from_rows(
    rows: np.ndarray | list[int] | tuple[int, ...],
    entity_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map one row of entity ids to an SCM input assignment."""
    indices = [int(part) for part in np.asarray(rows).reshape(-1)]
    return {
        variable: entity_vectors[index]
        for variable, index in zip(CANONICAL_INPUT_VARS, indices)
    }


def compute_states_for_rows(rows: np.ndarray | list[list[int]] | list[int]) -> dict[str, np.ndarray]:
    """Compute the intermediate booleans and final equality label."""
    arr = np.asarray(rows, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(CANONICAL_INPUT_VARS):
        raise ValueError(f"Expected rows shaped [N, 4], got {arr.shape}")

    w = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    z = arr[:, 3]
    wx = (w == x).astype(np.int64)
    yz = (y == z).astype(np.int64)
    o = (wx == yz).astype(np.int64)
    return {
        "W": w,
        "X": x,
        "Y": y,
        "Z": z,
        "WX": wx,
        "YZ": yz,
        "O": o,
    }


def compute_counterfactual_labels(
    base_states: dict[str, np.ndarray],
    source_states: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute SCM counterfactual outputs for each abstract variable swap."""
    return {
        "WX": (source_states["WX"] == base_states["YZ"]).astype(np.int64),
        "YZ": (base_states["WX"] == source_states["YZ"]).astype(np.int64),
    }


def sample_entity_rows(
    size: int,
    seed: int,
    num_entities: int = DEFAULT_NUM_ENTITIES,
) -> np.ndarray:
    """Sample random four-entity rows used for factual or pair-bank data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_entities, size=(size, len(CANONICAL_INPUT_VARS)), dtype=np.int64)


def rows_to_inputs_embeds(
    rows: np.ndarray | list[list[int]] | list[int],
    input_var_order: tuple[str, ...],
    entity_vectors: np.ndarray,
) -> torch.Tensor:
    """Pack entity-id rows into the concatenated continuous input format."""
    arr = np.asarray(rows, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(CANONICAL_INPUT_VARS):
        raise ValueError(f"Expected rows shaped [N, 4], got {arr.shape}")

    canonical_index = {var: index for index, var in enumerate(CANONICAL_INPUT_VARS)}
    ordered = np.stack([arr[:, canonical_index[var]] for var in input_var_order], axis=1)
    embeds = entity_vectors[ordered]
    return torch.tensor(embeds.reshape(arr.shape[0], -1), dtype=torch.float32)


def verify_scm_examples(problem: EqualityProblem, seed: int = 123, num_examples: int = 256) -> None:
    """Cross-check random rows against the symbolic SCM."""
    rows = sample_entity_rows(num_examples, seed, num_entities=problem.num_entities)
    states = compute_states_for_rows(rows)
    for index in range(rows.shape[0]):
        setting = problem.causal_model.run_forward(
            assignment_from_rows(rows[index], problem.entity_vectors)
        )
        for variable in ["WX", "YZ", "O"]:
            if int(setting[variable]) != int(states[variable][index]):
                raise AssertionError(
                    f"SCM mismatch at row={rows[index].tolist()} variable={variable}"
                )


def verify_counterfactual_labels_with_scm(
    problem: EqualityProblem,
    base_rows: np.ndarray,
    source_rows: np.ndarray,
    cf_labels_by_var: dict[str, np.ndarray],
) -> None:
    """Cross-check vectorized counterfactual labels against SCM interchange."""
    size = base_rows.shape[0]
    for index in range(size):
        base_assignment = assignment_from_rows(base_rows[index], problem.entity_vectors)
        source_assignment = assignment_from_rows(source_rows[index], problem.entity_vectors)
        for var in DEFAULT_TARGET_VARS:
            expected = int(
                problem.causal_model.run_interchange(base_assignment, {var: source_assignment})["O"]
            )
            actual = int(cf_labels_by_var[var][index])
            if expected != actual:
                raise AssertionError(
                    f"Counterfactual mismatch at index={index}, var={var}, expected={expected}, actual={actual}"
                )


def load_equality_problem(
    run_checks: bool = True,
    num_entities: int = DEFAULT_NUM_ENTITIES,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    seed: int = 0,
) -> EqualityProblem:
    """Build the hierarchical equality problem bundle."""
    entity_vectors = build_entity_vectors(
        num_entities=num_entities,
        embedding_dim=embedding_dim,
        seed=seed,
    )
    problem = EqualityProblem(
        causal_model=build_equality_causal_model(entity_vectors),
        input_var_order=CANONICAL_INPUT_VARS,
        entity_vectors=entity_vectors,
        embedding_dim=embedding_dim,
        num_entities=num_entities,
    )
    if run_checks:
        verify_scm_examples(problem)
    return problem
