"""IOI task definitions, dataset loading, and factual filtering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
from typing import Callable

from datasets import get_dataset_split_names, load_dataset
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from addition_experiment.runtime import resolve_device


DATASET_PATH = os.environ.get("IOI_DATASET_PATH", "mib-bench/ioi")
_DATASET_CONFIG_UNSET = object()


class IOICausalModel:
    """Causal model for IOI logit difference.

    Variables extracted from TextPrompt:
      STok (io_token / s_token): identity of the indirect object and subject names
      SPos (s_first): whether the subject appears first in the opening name pair

    Output: 0.048 + 0.768 * TokenSignal + 2.005 * PositionSignal
      TokenSignal  = +1 if IO is choices[0], -1 if IO is choices[1]
      PositionSignal = +1 if subject appears first in the name pair, -1 if second
    """

    _INTERCEPT: float = 0.048
    _TOKEN_COEF: float = 0.768
    _POSITION_COEF: float = 2.005

    def __init__(self) -> None:
        self.variables = (
            "io_token",
            "s_token",
            "s_first",
            "choices",
            "token_signal",
            "position_signal",
            "logit_diff",
            "answer_index",
            "answer",
            "raw_input",
        )

    def run_forward(self, input_dict: dict[str, object]) -> dict[str, object]:
        output = dict(input_dict)
        io_token = str(input_dict["io_token"])
        choices = list(input_dict["choices"])
        if choices[0] == io_token:
            io_choice_idx = 0
        elif len(choices) > 1 and choices[1] == io_token:
            io_choice_idx = 1
        else:
            raise ValueError(f"IO token {io_token!r} not in choices {choices}")
        token_signal = 1 if io_choice_idx == 0 else -1
        position_signal = 1 if bool(input_dict["s_first"]) else -1
        logit_diff = (
            self._INTERCEPT
            + self._TOKEN_COEF * token_signal
            + self._POSITION_COEF * position_signal
        )
        answer_index = 0 if logit_diff > 0 else 1
        output["token_signal"] = token_signal
        output["position_signal"] = position_signal
        output["logit_diff"] = float(logit_diff)
        output["answer_index"] = answer_index
        output["answer"] = str(choices[answer_index])
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
class IOIPairBank:
    """Tokenized base/source split for one IOI target variable."""

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
    choice_token_ids: torch.Tensor
    choice_token_variant_ids: torch.Tensor
    source_choice_token_ids: torch.Tensor
    source_choice_token_variant_ids: torch.Tensor
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


class IOIPairDataset(torch.utils.data.Dataset):
    """Dataset view for DAS training and evaluation."""

    def __init__(self, bank: IOIPairBank) -> None:
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
            "choice_token_ids": self.bank.choice_token_ids[index],
            "choice_token_variant_ids": self.bank.choice_token_variant_ids[index],
            "source_choice_token_ids": self.bank.source_choice_token_ids[index],
            "source_choice_token_variant_ids": self.bank.source_choice_token_variant_ids[index],
            "answer_token_id": self.bank.answer_token_ids[index],
            "base_answer_token_id": self.bank.base_answer_token_ids[index],
            "base_positions": {key: value[index] for key, value in self.bank.base_position_by_id.items()},
            "source_positions": {key: value[index] for key, value in self.bank.source_position_by_id.items()},
            "expected_answer_text": self.bank.expected_answer_texts[index],
        }


def parse_ioi_example(row: dict[str, object]) -> dict[str, object]:
    """Parse one HF IOI row into the causal-model input format."""
    prompt = str(row["prompt"])
    choices = list(row["choices"])
    answer_key = int(row["answerKey"])
    io_token = str(choices[answer_key])
    s_token = str(choices[1 - answer_key])
    io_match = re.search(rf"\b{re.escape(io_token)}\b", prompt)
    s_match = re.search(rf"\b{re.escape(s_token)}\b", prompt)
    if io_match is None or s_match is None:
        raise ValueError(
            f"Could not find names in prompt: {prompt!r}, io={io_token!r}, s={s_token!r}"
        )
    s_first = s_match.start() < io_match.start()
    return {
        "io_token": io_token,
        "s_token": s_token,
        "s_first": s_first,
        "choices": choices,
        "raw_input": prompt,
    }


def _find_name_token_index(prompt: str, name: str, tokenizer, occurrence: int = 0) -> int:
    """Return the token index of the n-th occurrence of name in prompt."""
    matches = list(re.finditer(rf"\b{re.escape(name)}\b", prompt))
    if occurrence >= len(matches):
        raise ValueError(
            f"Could not find occurrence {occurrence} of {name!r} in prompt: {prompt!r}"
        )
    char_end = matches[occurrence].end()
    substring = prompt[:char_end]
    token_ids = tokenizer(substring, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    return len(token_ids) - 1


def get_ioi_token_positions(tokenizer, causal_model: IOICausalModel) -> list[TokenPosition]:
    """Define IOI token positions: S1 (first subject), IO, S2 (second subject), last token."""

    def s1_position(input_dict: dict[str, object], tok) -> list[int]:
        return [_find_name_token_index(str(input_dict["raw_input"]), str(input_dict["s_token"]), tok, occurrence=0)]

    def io_position(input_dict: dict[str, object], tok) -> list[int]:
        return [_find_name_token_index(str(input_dict["raw_input"]), str(input_dict["io_token"]), tok, occurrence=0)]

    def s2_position(input_dict: dict[str, object], tok) -> list[int]:
        return [_find_name_token_index(str(input_dict["raw_input"]), str(input_dict["s_token"]), tok, occurrence=1)]

    def last_token(input_dict: dict[str, object], tok) -> list[int]:
        prompt = str(input_dict["raw_input"])
        token_ids = tok(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [len(token_ids) - 1]

    return [
        TokenPosition(s1_position, "s1"),
        TokenPosition(io_position, "io"),
        TokenPosition(s2_position, "s2"),
        TokenPosition(last_token, "last_token"),
    ]


def _load_ioi_counterfactual_rows(
    *,
    split: str,
    size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resolved_path = dataset_path or DATASET_PATH
    path_obj = Path(resolved_path)
    if path_obj.exists():
        split_file = path_obj / f"{split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Expected local IOI dataset file at {split_file}")
        dataset = load_dataset("json", data_files={split: str(split_file)}, split=split)
    else:
        dataset = load_dataset(resolved_path, split=split, token=token)
    if size is not None:
        dataset = dataset.select(range(min(int(size), len(dataset))))
    sample = dataset[0]
    counterfactual_names = [
        key
        for key in sample.keys()
        if key.endswith("_counterfactual") and key != "abc_counterfactual"
    ]
    datasets: dict[str, list[dict[str, object]]] = {}
    for counterfactual_name in counterfactual_names:
        dataset_name = counterfactual_name.replace("_counterfactual", f"_{split}")
        rows: list[dict[str, object]] = []
        for row in dataset:
            base_input = parse_ioi_example(row)
            counterfactual_row = row[counterfactual_name]
            source_input = parse_ioi_example(counterfactual_row)
            rows.append({"input": base_input, "counterfactual_inputs": [source_input]})
        datasets[dataset_name] = rows
    return datasets


def load_public_ioi_datasets(
    *,
    size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Load public train/validation/test IOI splits in the MIB counterfactual structure."""
    datasets: dict[str, list[dict[str, object]]] = {}
    resolved_path = dataset_path or DATASET_PATH
    path_obj = Path(resolved_path)
    if path_obj.exists():
        candidate_splits = tuple(
            split_file.stem for split_file in sorted(path_obj.glob("*.jsonl")) if split_file.stem
        )
        if not candidate_splits:
            raise FileNotFoundError(f"No .jsonl splits found under local dataset path {path_obj}")
    else:
        candidate_splits = tuple(get_dataset_split_names(resolved_path, token=hf_token))
    for split in candidate_splits:
        datasets.update(
            _load_ioi_counterfactual_rows(
                split=split,
                size=size,
                hf_token=hf_token,
                dataset_path=resolved_path,
            )
        )
    return datasets


def _infer_next_token_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    last_indices = torch.full(
        (logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device
    )
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices].argmax(dim=-1)


def normalize_answer_text(text: str) -> str:
    return str(text).strip()


def _encode_name_token(name: str, tokenizer) -> int:
    """Encode a name as a single vocabulary token, preferring the space-prefixed form."""
    name = normalize_answer_text(name)
    for candidate in (" " + name, name):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(
        f"Name {name!r} does not encode to a single token; IOI requires single-token names."
    )


def _encode_name_token_variants(name: str, tokenizer) -> tuple[int, int]:
    """Return (space-prefixed token ID, non-space token ID) for a name."""
    name = normalize_answer_text(name)
    variants = []
    for candidate in (" " + name, name):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            variants.append(int(ids[0]))
    if not variants:
        raise ValueError(f"Name {name!r} has no single-token encoding.")
    if len(variants) == 1:
        variants.append(variants[0])
    return (variants[0], variants[1])


def filter_correct_examples(
    *,
    model,
    tokenizer,
    causal_model: IOICausalModel,
    datasets_by_name: dict[str, list[dict[str, object]]],
    batch_size: int,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    """Keep only examples where the model predicts the factual IO token for the base prompt."""
    filtered: dict[str, list[dict[str, object]]] = {}
    for dataset_name, rows in datasets_by_name.items():
        print(f"[filter] dataset={dataset_name} total_rows={len(rows)}")
        prompts = [str(row["input"]["raw_input"]) for row in rows]
        expected_answers = [
            normalize_answer_text(str(causal_model.run_forward(row["input"])["answer"]))
            for row in rows
        ]
        expected_variant_ids = [_encode_name_token_variants(ans, tokenizer) for ans in expected_answers]
        keep_mask: list[bool] = []
        batch_starts = range(0, len(rows), batch_size)
        batch_iterator = batch_starts
        if tqdm is not None:
            batch_iterator = tqdm(
                batch_starts,
                desc=f"Filtering {dataset_name}",
                leave=False,
                total=(len(rows) + batch_size - 1) // batch_size,
            )
        for start in batch_iterator:
            end = min(start + batch_size, len(rows))
            encoded = tokenizer(
                prompts[start:end],
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            predicted_ids = _infer_next_token_ids(model, input_ids, attention_mask)
            for pred_id, expected, expected_variants in zip(
                predicted_ids.detach().cpu().tolist(),
                expected_answers[start:end],
                expected_variant_ids[start:end],
            ):
                decoded = normalize_answer_text(tokenizer.decode([int(pred_id)]))
                keep_mask.append(int(pred_id) in expected_variants or expected == decoded)
        filtered[dataset_name] = [row for row, keep in zip(rows, keep_mask) if keep]
        print(f"[filter] dataset={dataset_name} kept={len(filtered[dataset_name])}/{len(rows)}")
    return filtered


def _compute_row_change_masks(
    rows: list[dict[str, object]],
    causal_model: IOICausalModel,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[bool]]]:
    base_inputs = [row["input"] for row in rows]
    source_inputs = [row["counterfactual_inputs"][0] for row in rows]
    base_outputs = [causal_model.run_forward(b) for b in base_inputs]
    source_outputs = [causal_model.run_forward(s) for s in source_inputs]
    changed_masks = {
        "answer_index": [
            int(b["answer_index"]) != int(s["answer_index"])
            for b, s in zip(base_outputs, source_outputs)
        ],
    }
    return base_outputs, source_outputs, changed_masks


def build_pair_banks(
    *,
    tokenizer,
    causal_model: IOICausalModel,
    token_positions: list[TokenPosition],
    datasets_by_name: dict[str, list[dict[str, object]]],
    counterfactual_names: tuple[str, ...],
    target_vars: tuple[str, ...],
    split_seed: int = 0,
    train_pool_size: int | None = None,
    calibration_pool_size: int | None = None,
    test_pool_size: int | None = None,
    pooled_total_examples: int | None = None,
) -> tuple[dict[str, dict[str, IOIPairBank]], dict[str, object]]:
    """Build pooled train/calibration/test IOI banks with target-specific holdout sizing."""

    def make_bank(
        output_split: str,
        split_dataset_names: list[str],
        combined_rows: list[dict[str, object]],
    ) -> dict[str, IOIPairBank]:
        base_inputs = [row["input"] for row in combined_rows]
        source_inputs = [row["counterfactual_inputs"][0] for row in combined_rows]
        base_outputs = [causal_model.run_forward(b) for b in base_inputs]
        source_outputs = [causal_model.run_forward(s) for s in source_inputs]

        base_prompts = [str(b["raw_input"]) for b in base_inputs]
        source_prompts = [str(s["raw_input"]) for s in source_inputs]
        base_encoded = tokenizer(base_prompts, padding=True, return_tensors="pt", add_special_tokens=True)
        source_encoded = tokenizer(source_prompts, padding=True, return_tensors="pt", add_special_tokens=True)

        base_position_by_id = {
            tp.id: torch.tensor(
                [tp.resolve(b, tokenizer) for b in base_inputs],
                dtype=torch.long,
            )
            for tp in token_positions
        }
        source_position_by_id = {
            tp.id: torch.tensor(
                [tp.resolve(s, tokenizer) for s in source_inputs],
                dtype=torch.long,
            )
            for tp in token_positions
        }

        choice_token_ids = torch.tensor(
            [
                [_encode_name_token(str(b["choices"][j]), tokenizer) for j in range(2)]
                for b in base_inputs
            ],
            dtype=torch.long,
        )
        choice_token_variant_ids = torch.tensor(
            [
                [_encode_name_token_variants(str(b["choices"][j]), tokenizer) for j in range(2)]
                for b in base_inputs
            ],
            dtype=torch.long,
        )
        source_choice_token_ids = torch.tensor(
            [
                [_encode_name_token(str(s["choices"][j]), tokenizer) for j in range(2)]
                for s in source_inputs
            ],
            dtype=torch.long,
        )
        source_choice_token_variant_ids = torch.tensor(
            [
                [_encode_name_token_variants(str(s["choices"][j]), tokenizer) for j in range(2)]
                for s in source_inputs
            ],
            dtype=torch.long,
        )

        answer_token_ids = torch.tensor(
            [_encode_name_token(normalize_answer_text(str(so["answer"])), tokenizer) for so in source_outputs],
            dtype=torch.long,
        )
        base_answer_token_ids = torch.tensor(
            [_encode_name_token(normalize_answer_text(str(bo["answer"])), tokenizer) for bo in base_outputs],
            dtype=torch.long,
        )

        answer_label_indices = torch.tensor(
            [int(so["answer_index"]) for so in source_outputs],
            dtype=torch.long,
        )
        changed_answer = torch.tensor(
            [int(bo["answer_index"]) != int(so["answer_index"]) for bo, so in zip(base_outputs, source_outputs)],
            dtype=torch.bool,
        )

        banks: dict[str, IOIPairBank] = {}
        for target_var in target_vars:
            if target_var == "answer_index":
                labels = answer_label_indices
                changed_mask = changed_answer
            else:
                raise ValueError(f"Unsupported IOI target variable {target_var!r}")
            banks[target_var] = IOIPairBank(
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
                choice_token_ids=choice_token_ids,
                choice_token_variant_ids=choice_token_variant_ids,
                source_choice_token_ids=source_choice_token_ids,
                source_choice_token_variant_ids=source_choice_token_variant_ids,
                answer_token_ids=answer_token_ids,
                base_answer_token_ids=base_answer_token_ids,
                changed_mask=changed_mask,
                expected_answer_texts=[
                    normalize_answer_text(str(so["answer"])) for so in source_outputs
                ],
            )
        return banks

    banks_by_split: dict[str, dict[str, IOIPairBank]] = {"train": {}, "calibration": {}, "test": {}}
    pooled_dataset_names: list[str] = []
    pooled_rows: list[dict[str, object]] = []
    for dataset_name in sorted(datasets_by_name):
        counterfactual_name, _, _split_name = dataset_name.rpartition("_")
        if counterfactual_name in counterfactual_names:
            pooled_dataset_names.append(dataset_name)
            pooled_rows.extend(datasets_by_name[dataset_name])
    if not pooled_rows:
        raise ValueError("No IOI rows found for pooled bank construction")

    rng = random.Random(int(split_seed))
    shuffled_rows = list(pooled_rows)
    rng.shuffle(shuffled_rows)
    if pooled_total_examples is not None:
        shuffled_rows = shuffled_rows[: min(int(pooled_total_examples), len(shuffled_rows))]
    total = len(shuffled_rows)

    resolved_train_pool_size = total if train_pool_size is None else int(train_pool_size)
    resolved_calibration_pool_size = 0 if calibration_pool_size is None else int(calibration_pool_size)
    resolved_test_pool_size = 0 if test_pool_size is None else int(test_pool_size)
    if resolved_train_pool_size < 0 or resolved_calibration_pool_size < 0 or resolved_test_pool_size < 0:
        raise ValueError("train_pool_size, calibration_pool_size, and test_pool_size must be non-negative")
    if resolved_train_pool_size > total:
        raise ValueError(
            f"Requested train_pool_size={resolved_train_pool_size}, but only {total} IOI rows are available"
        )

    train_rows = shuffled_rows[:resolved_train_pool_size]
    holdout_candidate_rows = shuffled_rows[resolved_train_pool_size:]
    if not train_rows:
        raise ValueError("No IOI rows found for pooled train split")

    train_rng = random.Random(f"{int(split_seed)}:train:shared")
    shared_train_rows = list(train_rows)
    train_rng.shuffle(shared_train_rows)
    train_banks = make_bank("train", pooled_dataset_names, shared_train_rows)
    for target_var in target_vars:
        banks_by_split["train"][target_var] = train_banks[target_var]

    _base_outputs, _source_outputs, holdout_changed_masks = _compute_row_change_masks(
        holdout_candidate_rows, causal_model
    )
    for target_var in target_vars:
        changed_mask = holdout_changed_masks[target_var]
        positive_rows = [row for row, changed in zip(holdout_candidate_rows, changed_mask) if changed]
        local_rng = random.Random(f"{int(split_seed)}:holdout:{target_var}")
        local_rng.shuffle(positive_rows)
        required = resolved_calibration_pool_size + resolved_test_pool_size
        if len(positive_rows) < required:
            raise ValueError(
                f"Requested calibration_pool_size={resolved_calibration_pool_size} and "
                f"test_pool_size={resolved_test_pool_size} for target_var={target_var}, "
                f"but only {len(positive_rows)} sensitive rows are available after train allocation"
            )
        calibration_rows = positive_rows[:resolved_calibration_pool_size]
        test_rows = positive_rows[resolved_calibration_pool_size:required]
        if resolved_calibration_pool_size > 0:
            banks_by_split["calibration"][target_var] = make_bank(
                "calibration", pooled_dataset_names, calibration_rows
            )[target_var]
        if resolved_test_pool_size > 0:
            banks_by_split["test"][target_var] = make_bank(
                "test", pooled_dataset_names, test_rows
            )[target_var]

    if resolved_calibration_pool_size == 0:
        banks_by_split["calibration"] = {}
    if resolved_test_pool_size == 0:
        banks_by_split["test"] = {}

    metadata = {
        split: {target_var: bank.metadata() for target_var, bank in banks.items()}
        for split, banks in banks_by_split.items()
    }
    return banks_by_split, metadata


def load_filtered_ioi_pipeline(
    *,
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
    dataset_size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
) -> tuple[object, object, IOICausalModel, list[TokenPosition], dict[str, list[dict[str, object]]]]:
    """Load a causal LM, set up the IOI task, and filter to factually-correct base examples."""
    import transformers

    torch_device = resolve_device(device)
    print(f"[load] device={torch_device} model={model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if torch_device.type in {"cuda", "mps"} else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=hf_token,
        attn_implementation="eager",
    )
    model.to(torch_device)
    model.eval()
    print(f"[load] model and tokenizer ready dtype={dtype}")

    causal_model = IOICausalModel()
    token_positions = get_ioi_token_positions(tokenizer, causal_model)
    print(f"[load] token_positions={[tp.id for tp in token_positions]}")

    resolved_path = dataset_path or DATASET_PATH
    print(f"[load] loading IOI datasets path={resolved_path} size_cap={dataset_size}")
    public_datasets = load_public_ioi_datasets(
        size=dataset_size,
        hf_token=hf_token,
        dataset_path=resolved_path,
    )
    print(f"[load] loaded datasets={sorted(public_datasets.keys())}")
    print("[load] starting factual filtering")
    filtered_datasets = filter_correct_examples(
        model=model,
        tokenizer=tokenizer,
        causal_model=causal_model,
        datasets_by_name=public_datasets,
        batch_size=batch_size,
        device=torch_device,
    )
    print("[load] factual filtering complete")
    return model, tokenizer, causal_model, token_positions, filtered_datasets
