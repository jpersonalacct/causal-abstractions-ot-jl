"""DAS search and evaluation on hierarchical-equality pair-bank interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import _env  # noqa: F401

from pyvene import RotatedSpaceIntervention
from torch.utils.data import DataLoader

from variable_width_mlp import VariableWidthMLPForClassification, logits_from_output

from .constants import DEFAULT_TARGET_VARS
from .metrics import metrics_from_logits
from .pair_bank import PairBank, PairBankVariableDataset
from .pyvene_utils import DASSearchSpec, build_intervenable, run_intervenable_logits


@dataclass(frozen=True)
class DASConfig:
    """Hyperparameters controlling DAS training, selection, and evaluation."""

    batch_size: int = 128
    max_epochs: int = 1
    learning_rate: float = 1e-3
    subspace_dims: tuple[int, ...] | None = None
    search_layers: tuple[int, ...] | None = None
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS
    plateau_patience: int = 2
    plateau_rel_delta: float = 5e-3
    min_epochs: int = 1
    verbose: bool = True


def iter_search_specs(
    model: VariableWidthMLPForClassification,
    config: DASConfig,
) -> list[DASSearchSpec]:
    """Enumerate DAS layer and subspace-size candidates to test."""
    layers = (
        list(range(model.config.n_layer))
        if config.search_layers is None
        else [int(layer) for layer in config.search_layers]
    )
    specs = []
    for layer in layers:
        width = int(model.config.hidden_dims[layer])
        subspace_dims = config.subspace_dims
        if subspace_dims is None:
            subspace_dims = tuple(range(1, width + 1))
        for subspace_dim in subspace_dims:
            if int(subspace_dim) <= width:
                specs.append(
                    DASSearchSpec(
                        layer=layer,
                        subspace_dim=int(subspace_dim),
                        component=f"h[{layer}].output",
                    )
                )
    return specs


def evaluate_rotated_intervention(
    intervenable,
    dataset: PairBankVariableDataset,
    spec: DASSearchSpec,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one trained DAS intervention on a dataset split."""
    logits_all = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        logits = run_intervenable_logits(
            intervenable=intervenable,
            base_inputs=batch["input_ids"],
            source_inputs=batch["source_input_ids"],
            subspace_dims=spec.subspace_dims,
            position=spec.position,
            batch_size=batch_size,
            device=device,
        )
        logits_all.append(logits)
        labels_all.append(batch["labels"].to(torch.long).view(-1))
    return metrics_from_logits(torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0))


def train_rotated_intervention(
    intervenable,
    dataset: PairBankVariableDataset,
    spec: DASSearchSpec,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    plateau_patience: int,
    plateau_rel_delta: float,
    min_epochs: int,
) -> list[float]:
    """Train the DAS rotation parameters for one candidate intervention."""
    optimizer_parameters = []
    for intervention in intervenable.interventions.values():
        if hasattr(intervention, "rotate_layer"):
            optimizer_parameters.append({"params": intervention.rotate_layer.parameters()})
    if not optimizer_parameters:
        raise RuntimeError("No rotate_layer parameters found for DAS intervention")

    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    losses = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_loss = None
    plateau_steps = 0

    for epoch_index in range(max_epochs):
        epoch_losses = []
        for batch in loader:
            device_base = batch["input_ids"].to(device)
            device_source = batch["source_input_ids"].to(device)
            labels = batch["labels"].to(device).view(-1)
            batch_size_now = device_base.shape[0]

            positions = [[spec.position]] * batch_size_now
            subspaces = [spec.subspace_dims] * batch_size_now
            base_batch = device_base.unsqueeze(1) if device_base.ndim == 2 else device_base
            source_batch = device_source.unsqueeze(1) if device_source.ndim == 2 else device_source[:, :1, :]

            _, cf_output = intervenable(
                {"inputs_embeds": base_batch},
                [{"inputs_embeds": source_batch}],
                {"sources->base": ([positions], [positions])},
                subspaces=[subspaces],
            )
            logits = logits_from_output(cf_output)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(epoch_loss)
        if best_loss is None:
            improved = True
        else:
            relative_threshold = float(best_loss) * (1.0 - float(plateau_rel_delta))
            improved = epoch_loss < relative_threshold
        if improved:
            best_loss = epoch_loss
            plateau_steps = 0
        else:
            plateau_steps += 1
        if (
            plateau_patience > 0
            and epoch_index + 1 >= int(min_epochs)
            and plateau_steps >= int(plateau_patience)
        ):
            break
    return losses


def choose_better_result(candidate: dict[str, object], incumbent: dict[str, object] | None) -> bool:
    """Decide whether a candidate beats the current DAS incumbent."""
    if incumbent is None:
        return True
    candidate_key = (float(candidate["selection_exact_acc"]), float(candidate["selection_mean_shared_digits"]))
    incumbent_key = (float(incumbent["selection_exact_acc"]), float(incumbent["selection_mean_shared_digits"]))
    return candidate_key > incumbent_key


def run_das_search_for_variable(
    model: VariableWidthMLPForClassification,
    variable: str,
    train_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device,
    config: DASConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Search DAS candidates for one abstract variable and keep the best one."""
    specs = iter_search_specs(model, config)
    train_dataset = PairBankVariableDataset(train_bank, variable)
    calibration_dataset = PairBankVariableDataset(calibration_bank, variable)

    best_record = None
    best_intervenable = None
    best_spec = None
    all_records = []
    if config.verbose:
        print(
            f"DAS [{variable}] "
            f"| candidates={len(specs)} "
            f"| train_examples={len(train_dataset)} "
            f"| calibration_examples={len(calibration_dataset)} "
            f"| holdout_examples={holdout_bank.size}"
        )
    for index, spec in enumerate(specs, start=1):
        intervention = RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[spec.layer]))
        intervenable = build_intervenable(
            model=model,
            layer=spec.layer,
            component=spec.component,
            intervention=intervention,
            device=device,
            unit=spec.unit,
            max_units=spec.max_units,
            freeze_model=True,
            freeze_intervention=False,
            use_fast=False,
        )
        loss_history = train_rotated_intervention(
            intervenable=intervenable,
            dataset=train_dataset,
            spec=spec,
            max_epochs=config.max_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            device=device,
            plateau_patience=config.plateau_patience,
            plateau_rel_delta=config.plateau_rel_delta,
            min_epochs=config.min_epochs,
        )
        calibration_metrics = evaluate_rotated_intervention(
            intervenable=intervenable,
            dataset=calibration_dataset,
            spec=spec,
            batch_size=config.batch_size,
            device=device,
        )
        record = {
            "method": "das",
            "variable": variable,
            "split": calibration_bank.split,
            "seed": calibration_bank.seed,
            "site_label": spec.label,
            "layer": spec.layer,
            "subspace_dim": spec.subspace_dim,
            "selection_exact_acc": calibration_metrics["exact_acc"],
            "selection_mean_shared_digits": calibration_metrics["mean_shared_digits"],
            "calibration_exact_acc": calibration_metrics["exact_acc"],
            "calibration_mean_shared_digits": calibration_metrics["mean_shared_digits"],
            "train_epochs_ran": len(loss_history),
            "train_loss_history": loss_history,
        }
        all_records.append(record)
        is_better = choose_better_result(record, best_record)
        if config.verbose:
            status = "new best" if is_better else "candidate"
            print(
                f"DAS [{variable}] {status} {index}/{len(specs)} "
                f"| site={spec.label} "
                f"| epochs={len(loss_history)} "
                f"| train_loss={loss_history[-1]:.4f} "
                f"| calibration_exact={calibration_metrics['exact_acc']:.4f}"
            )
        if is_better:
            best_record = record
            best_intervenable = intervenable
            best_spec = spec

    if best_record is None or best_intervenable is None or best_spec is None:
        raise RuntimeError(f"Failed to select a DAS candidate for {variable}")

    holdout_dataset = PairBankVariableDataset(holdout_bank, variable)
    holdout_metrics = evaluate_rotated_intervention(
        intervenable=best_intervenable,
        dataset=holdout_dataset,
        spec=best_spec,
        batch_size=config.batch_size,
        device=device,
    )
    result_record = {
        **best_record,
        "split": holdout_bank.split,
        "seed": holdout_bank.seed,
        **holdout_metrics,
    }
    return result_record, all_records


def run_das_pipeline(
    model: VariableWidthMLPForClassification,
    train_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device | str,
    config: DASConfig,
) -> dict[str, object]:
    """Run DAS for every abstract variable on shared pair-bank splits."""
    device = torch.device(device)
    results = []
    search_records = {}
    for variable in config.target_vars:
        best_record, all_records = run_das_search_for_variable(
            model=model,
            variable=variable,
            train_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=holdout_bank,
            device=device,
            config=config,
        )
        results.append(best_record)
        search_records[variable] = all_records

    return {
        "train_bank": train_bank.metadata(),
        "calibration_bank": calibration_bank.metadata(),
        "holdout_bank": holdout_bank.metadata(),
        "target_vars": list(config.target_vars),
        "training_stopping_rule": {
            "type": "plateau_on_train_loss",
            "max_epochs": config.max_epochs,
            "min_epochs": config.min_epochs,
            "plateau_rel_delta": config.plateau_rel_delta,
            "plateau_patience": config.plateau_patience,
        },
        "search_records": search_records,
        "results": results,
    }
