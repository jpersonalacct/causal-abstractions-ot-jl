"""MCQA task definitions, dataset loading, and factual filtering."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import re
from typing import Callable

from datasets import load_dataset
import torch

from .runtime import resolve_device


DATASET_PATH = "mib-bench/copycolors_mcqa"
DATASET_NAME = "4_answer_choices"

CANONICAL_ANSWER_STRINGS = (" A", " B", " C", " D")
CANONICAL_ANSWER_LABELS = ("A", "B", "C", "D")


class MCQACausalModel:
    """Small self-contained copy of the Simple MCQA causal model logic."""

    def __init__(self) -> None:
        self.variables = (
            "question",
            "raw_input",
            "symbol0",
            "symbol1",
            "symbol2",
            "symbol3",
            "choice0",
            "choice1",
            "choice2",
            "choice3",
            "answer_pointer",
            "answer",
            "raw_output",
        )

    def run_forward(self, input_dict: dict[str, object]) -> dict[str, object]:
        output = dict(input_dict)
        question = tuple(output["question"])
        choices = [str(output[f"choice{index}"]) for index in range(4)]
        symbols = [str(output[f"symbol{index}"]) for index in range(4)]
        pointer = None
        for index, choice in enumerate(choices):
            if choice == question[0]:
                pointer = index
                break
        if pointer is None:
            raise ValueError(f"Could not resolve answer_pointer from question={question} and choices={choices}")
        answer = " " + symbols[pointer]
        output["answer_pointer"] = int(pointer)
        output["answer"] = answer
        output["raw_output"] = answer
        return output


@dataclass(frozen=True)
class TokenPosition:
    """Minimal task-local token-position descriptor."""

    resolver: Callable[[dict[str, object], object], list[int]]
    id: str

    def resolve(self, input_dict: dict[str, object], tokenizer) -> int:
        positions = self.resolver(input_dict, tokenizer)
        if not positions:
            raise ValueError(f"Token position {self.id} returned no indices")
        return int(positions[0])


@dataclass(frozen=True)
class MCQAPairBank:
    """Tokenized base/source split for one MCQA target variable."""

    split: str
    target_var: str
    dataset_names: tuple[str, ...]
    base_input_ids: torch.Tensor
    base_attention_mask: torch.Tensor
    source_input_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    labels: torch.Tensor
    base_inputs: list[dict[str, object]]
    source_inputs: list[dict[str, object]]
    base_outputs: list[dict[str, object]]
    source_outputs: list[dict[str, object]]
    base_position_by_id: dict[str, torch.Tensor]
    source_position_by_id: dict[str, torch.Tensor]
    symbol_token_ids: torch.Tensor
    source_symbol_token_ids: torch.Tensor
    canonical_answer_token_ids: torch.Tensor
    answer_token_ids: torch.Tensor
    base_answer_token_ids: torch.Tensor
    changed_mask: torch.Tensor
    expected_answer_texts: list[str]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def metadata(self) -> dict[str, object]:
        return {
            "split": self.split,
            "target_var": self.target_var,
            "size": self.size,
            "dataset_names": list(self.dataset_names),
            "changed_count": int(self.changed_mask.sum().item()),
            "changed_rate": float(self.changed_mask.float().mean().item()) if self.size else 0.0,
        }


class MCQAPairDataset(torch.utils.data.Dataset):
    """Dataset view for DAS training and evaluation."""

    def __init__(self, bank: MCQAPairBank):
        self.bank = bank

    def __len__(self) -> int:
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "base_input_ids": self.bank.base_input_ids[index],
            "base_attention_mask": self.bank.base_attention_mask[index],
            "source_input_ids": self.bank.source_input_ids[index],
            "source_attention_mask": self.bank.source_attention_mask[index],
            "labels": self.bank.labels[index],
            "symbol_token_ids": self.bank.symbol_token_ids[index],
            "source_symbol_token_ids": self.bank.source_symbol_token_ids[index],
            "answer_token_id": self.bank.answer_token_ids[index],
            "base_answer_token_id": self.bank.base_answer_token_ids[index],
            "base_positions": {key: value[index] for key, value in self.bank.base_position_by_id.items()},
            "source_positions": {key: value[index] for key, value in self.bank.source_position_by_id.items()},
            "expected_answer_text": self.bank.expected_answer_texts[index],
        }


def parse_mcqa_example(row: dict[str, object]) -> dict[str, object]:
    """Parse one HF MCQA row into the copied causal-model input format."""
    prompt_str = str(row.get("prompt", ""))
    question_text = prompt_str
    if " is " in question_text:
        noun, color = question_text.split(" is ", 1)
    elif " are " in question_text:
        noun, color = question_text.split(" are ", 1)
    else:
        raise ValueError(f"Could not parse MCQA question text from prompt: {prompt_str}")
    noun = noun.strip().lower()
    color = color.split(".", 1)[0].strip().lower()
    variables_dict: dict[str, object] = {
        "question": (color, noun),
        "raw_input": prompt_str,
    }
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    for index, label in enumerate(labels):
        variables_dict[f"symbol{index}"] = str(label)
        variables_dict[f"choice{index}"] = str(texts[index])
    return variables_dict


def _find_correct_symbol_index(input_dict: dict[str, object], tokenizer, causal_model: MCQACausalModel) -> list[int]:
    output = causal_model.run_forward(input_dict)
    pointer = int(output["answer_pointer"])
    correct_symbol = str(output[f"symbol{pointer}"])
    prompt = str(input_dict["raw_input"])
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))
    symbol_match = None
    for match in matches:
        if prompt[match.start() : match.end()] == correct_symbol:
            symbol_match = match
            break
    if symbol_match is None:
        raise ValueError(f"Could not find correct symbol {correct_symbol} in prompt: {prompt}")
    substring = prompt[: symbol_match.end()]
    tokenized = tokenizer(substring, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    return [len(tokenized) - 1]


def get_token_positions(tokenizer, causal_model: MCQACausalModel) -> list[TokenPosition]:
    """Copied token-position logic for Simple MCQA."""

    def correct_symbol(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        return _find_correct_symbol_index(input_dict, current_tokenizer, causal_model)

    def correct_symbol_period(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        return [correct_symbol(input_dict, current_tokenizer)[0] + 1]

    def last_token(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        prompt = str(input_dict["raw_input"])
        tokenized = current_tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [len(tokenized) - 1]

    return [
        TokenPosition(correct_symbol, "correct_symbol"),
        TokenPosition(correct_symbol_period, "correct_symbol_period"),
        TokenPosition(last_token, "last_token"),
    ]


def _load_counterfactual_rows(
    *,
    split: str,
    size: int | None = None,
    hf_token: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=split, token=token)
    if size is not None:
        dataset = dataset.select(range(min(int(size), len(dataset))))
    sample = dataset[0]
    counterfactual_names = [
        key
        for key in sample.keys()
        if key.endswith("_counterfactual") and "noun" not in key and "color" not in key and "symbol" not in key
    ]
    datasets: dict[str, list[dict[str, object]]] = {}
    for counterfactual_name in counterfactual_names:
        dataset_name = counterfactual_name.replace("_counterfactual", f"_{split}")
        rows: list[dict[str, object]] = []
        for row in dataset:
            base_input = parse_mcqa_example(row)
            counterfactual_row = row[counterfactual_name]
            source_input = parse_mcqa_example(counterfactual_row)
            rows.append(
                {
                    "input": base_input,
                    "counterfactual_inputs": [source_input],
                }
            )
        datasets[dataset_name] = rows
    return datasets


def load_public_mcqa_datasets(*, size: int | None = None, hf_token: str | None = None) -> dict[str, list[dict[str, object]]]:
    """Load public train/validation/test MCQA splits in the copied MIB structure."""
    datasets: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "validation", "test"):
        datasets.update(_load_counterfactual_rows(split=split, size=size, hf_token=hf_token))
    return datasets


def _infer_next_token_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = outputs.logits
    last_indices = attention_mask.sum(dim=1).to(torch.long) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices].argmax(dim=-1)


def filter_correct_examples(
    *,
    model,
    tokenizer,
    causal_model: MCQACausalModel,
    datasets_by_name: dict[str, list[dict[str, object]]],
    batch_size: int,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    """Keep only examples where Gemma predicts the factual answer token."""
    filtered: dict[str, list[dict[str, object]]] = {}
    for dataset_name, rows in datasets_by_name.items():
        prompts = [str(row["input"]["raw_input"]) for row in rows]
        expected_answers = [str(causal_model.run_forward(row["input"])["answer"]) for row in rows]
        keep_mask: list[bool] = []
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_prompts = prompts[start:end]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            predicted_ids = _infer_next_token_ids(model, input_ids, attention_mask)
            for predicted_id, expected in zip(predicted_ids.detach().cpu().tolist(), expected_answers[start:end]):
                decoded = tokenizer.decode([int(predicted_id)])
                keep_mask.append(expected in decoded)
        filtered[dataset_name] = [row for row, keep in zip(rows, keep_mask) if keep]
    return filtered


def _validate_answer_tokenization(tokenizer) -> torch.Tensor:
    token_ids = []
    for token_text in CANONICAL_ANSWER_STRINGS:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Expected {token_text!r} to map to a single token, but got token ids {ids}. "
                "MCQA v1 requires single-token answer letters."
            )
        token_ids.append(int(ids[0]))
    return torch.tensor(token_ids, dtype=torch.long)


def _encode_symbol_token(symbol: str, tokenizer) -> int:
    ids = tokenizer.encode(" " + str(symbol), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected symbol {symbol!r} to map to one token, but got ids {ids}")
    return int(ids[0])


def build_pair_banks(
    *,
    tokenizer,
    causal_model: MCQACausalModel,
    token_positions: list[TokenPosition],
    datasets_by_name: dict[str, list[dict[str, object]]],
    counterfactual_names: tuple[str, ...],
    target_vars: tuple[str, ...],
) -> tuple[dict[str, dict[str, MCQAPairBank]], dict[str, object]]:
    """Build train/calibration/test banks for both target variables."""
    canonical_answer_token_ids = _validate_answer_tokenization(tokenizer)
    split_map = {
        "train": "train",
        "validation": "calibration",
        "test": "test",
    }
    banks_by_split: dict[str, dict[str, MCQAPairBank]] = {split: {} for split in split_map.values()}
    for hf_split, output_split in split_map.items():
        split_dataset_names = [
            f"{counterfactual_name}_{hf_split}"
            for counterfactual_name in counterfactual_names
            if f"{counterfactual_name}_{hf_split}" in datasets_by_name
        ]
        combined_rows: list[dict[str, object]] = []
        for dataset_name in split_dataset_names:
            combined_rows.extend(datasets_by_name[dataset_name])
        if not combined_rows:
            raise ValueError(f"No MCQA rows found for split {hf_split}")

        base_inputs = [row["input"] for row in combined_rows]
        source_inputs = [row["counterfactual_inputs"][0] for row in combined_rows]
        base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
        source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]

        base_prompts = [str(base_input["raw_input"]) for base_input in base_inputs]
        source_prompts = [str(source_input["raw_input"]) for source_input in source_inputs]
        base_encoded = tokenizer(base_prompts, padding=True, return_tensors="pt", add_special_tokens=True)
        source_encoded = tokenizer(source_prompts, padding=True, return_tensors="pt", add_special_tokens=True)

        base_position_by_id = {
            token_position.id: torch.tensor(
                [token_position.resolve(base_input, tokenizer) for base_input in base_inputs],
                dtype=torch.long,
            )
            for token_position in token_positions
        }
        source_position_by_id = {
            token_position.id: torch.tensor(
                [token_position.resolve(source_input, tokenizer) for source_input in source_inputs],
                dtype=torch.long,
            )
            for token_position in token_positions
        }
        symbol_token_ids = torch.tensor(
            [
                [_encode_symbol_token(str(base_input[f"symbol{index}"]), tokenizer) for index in range(4)]
                for base_input in base_inputs
            ],
            dtype=torch.long,
        )
        source_symbol_token_ids = torch.tensor(
            [
                [_encode_symbol_token(str(source_input[f"symbol{index}"]), tokenizer) for index in range(4)]
                for source_input in source_inputs
            ],
            dtype=torch.long,
        )
        answer_token_ids = torch.tensor(
            [_encode_symbol_token(str(source_output["answer"]).strip(), tokenizer) for source_output in source_outputs],
            dtype=torch.long,
        )
        base_answer_token_ids = torch.tensor(
            [_encode_symbol_token(str(base_output["answer"]).strip(), tokenizer) for base_output in base_outputs],
            dtype=torch.long,
        )
        answer_pointer_labels = torch.tensor(
            [int(source_output["answer_pointer"]) for source_output in source_outputs],
            dtype=torch.long,
        )
        changed_pointer = torch.tensor(
            [
                int(base_output["answer_pointer"]) != int(source_output["answer_pointer"])
                for base_output, source_output in zip(base_outputs, source_outputs)
            ],
            dtype=torch.bool,
        )
        changed_answer = torch.tensor(
            [
                str(base_output["answer"]) != str(source_output["answer"])
                for base_output, source_output in zip(base_outputs, source_outputs)
            ],
            dtype=torch.bool,
        )

        for target_var in target_vars:
            if target_var == "answer_pointer":
                labels = answer_pointer_labels
                changed_mask = changed_pointer
            elif target_var == "answer":
                labels = answer_pointer_labels
                changed_mask = changed_answer
            else:
                raise ValueError(f"Unsupported MCQA target variable {target_var}")
            banks_by_split[output_split][target_var] = MCQAPairBank(
                split=output_split,
                target_var=target_var,
                dataset_names=tuple(split_dataset_names),
                base_input_ids=base_encoded["input_ids"].to(torch.long),
                base_attention_mask=base_encoded["attention_mask"].to(torch.long),
                source_input_ids=source_encoded["input_ids"].to(torch.long),
                source_attention_mask=source_encoded["attention_mask"].to(torch.long),
                labels=labels,
                base_inputs=base_inputs,
                source_inputs=source_inputs,
                base_outputs=base_outputs,
                source_outputs=source_outputs,
                base_position_by_id=base_position_by_id,
                source_position_by_id=source_position_by_id,
                symbol_token_ids=symbol_token_ids,
                source_symbol_token_ids=source_symbol_token_ids,
                canonical_answer_token_ids=canonical_answer_token_ids,
                answer_token_ids=answer_token_ids,
                base_answer_token_ids=base_answer_token_ids,
                changed_mask=changed_mask,
                expected_answer_texts=[str(source_output["answer"]) for source_output in source_outputs],
            )
    metadata = {
        split: {target_var: bank.metadata() for target_var, bank in banks.items()}
        for split, banks in banks_by_split.items()
    }
    return banks_by_split, metadata


def load_filtered_mcqa_pipeline(
    *,
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
    dataset_size: int | None = None,
    hf_token: str | None = None,
) -> tuple[object, object, MCQACausalModel, list[TokenPosition], dict[str, list[dict[str, object]]]]:
    """Load Gemma-2-2B, copy the MCQA task setup, and filter to correct examples."""
    import transformers

    torch_device = resolve_device(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if torch_device.type in {"cuda", "mps"} else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=hf_token)
    model.to(torch_device)
    model.eval()
    causal_model = MCQACausalModel()
    token_positions = get_token_positions(tokenizer, causal_model)
    public_datasets = load_public_mcqa_datasets(size=dataset_size, hf_token=hf_token)
    filtered_datasets = filter_correct_examples(
        model=model,
        tokenizer=tokenizer,
        causal_model=causal_model,
        datasets_by_name=public_datasets,
        batch_size=batch_size,
        device=torch_device,
    )
    return model, tokenizer, causal_model, token_positions, filtered_datasets
