# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preserve sparse per-sample reward details through diffusion validation."""

from collections.abc import Mapping

from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.reward import extract_reward


def _aligned_values(values, expected, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"Validation field {name!r} must contain one value per sample")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(f"Validation field {name!r} must contain one value per sample") from exc
    if len(values) != expected:
        raise ValueError(f"Validation field {name!r} has {len(values)} values; expected {expected}")
    return values


def extract_validation_reward(data):
    """Recover streaming reward details without changing the reward tensor.

    Reward managers already expose flattened columns through ``extract_reward``.
    TransferQueue instead retains one ``reward_extra_info`` mapping per sample.
    Preserve the manager contract when it is available, and otherwise unpack
    those rows with explicit ``None`` entries for absent dimensions.
    """
    reward, extras = extract_reward(data)
    if extras:
        return reward, extras
    nested = data.non_tensor_batch.get("reward_extra_info")
    if nested is None:
        return reward, extras
    rows = _aligned_values(nested, len(data), "reward_extra_info")
    keys = {}
    for row in rows:
        if row is not None and not isinstance(row, Mapping):
            raise ValueError("Validation reward_extra_info rows must be mappings or None")
        if row is not None:
            keys.update(dict.fromkeys(row))
    extras = {key: [row.get(key) if row is not None else None for row in rows] for key in keys}
    return reward, extras


def append_validation_reward_extra(accumulator, extras, *, previous_samples, batch_size):
    """Append aligned detail columns; the caller separately owns total ``reward``.

    Backfill newly seen dimensions and pad absent dimensions with ``None`` so
    validation media dumps retain one entry per generated sample. Validate all
    lengths before mutating the accumulator.
    """
    for name, value in (("previous_samples", previous_samples), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    incoming = {
        key: _aligned_values(values, batch_size, key) for key, values in extras.items() if key != "reward"
    }
    existing = {
        key: _aligned_values(values, previous_samples, key)
        for key, values in accumulator.items()
        if key != "reward"
    }
    for key in dict.fromkeys((*existing, *incoming)):
        accumulator[key] = existing.get(key, [None] * previous_samples) + incoming.get(key, [None] * batch_size)


def process_validation_reward_metrics(data_sources, sample_uids, infos):
    """Delegate statistics to verl after selecting the rows each metric has.

    Only absent values (``None``) are removed; NaN and infinity remain visible.
    Data-source and prompt identities are kept, including repeated generations.
    If predictions are incomplete, ordinary statistics still use all available
    scores, while majority statistics use only prompt groups with a prediction
    for every retained score. Missing predictions never become a vote category.
    """
    sources = list(data_sources)
    uids = _aligned_values(sample_uids, len(sources), "sample_uids")
    columns = {key: _aligned_values(values, len(sources), key) for key, values in infos.items()}
    if all(value is not None for values in columns.values() for value in values):
        return process_validation_metrics(sources, uids, columns)
    masks = {}
    for key, values in columns.items():
        indices = tuple(index for index, value in enumerate(values) if value is not None)
        if indices:
            masks.setdefault(indices, []).append(key)

    result = {}

    def merge(metrics, names, *, majority_only=False):
        for source, variables in metrics.items():
            for name in names:
                if name not in variables:
                    continue
                values = variables[name]
                if majority_only:
                    values = {key: value for key, value in values.items() if key.startswith("maj@")}
                if values:
                    result.setdefault(source, {}).setdefault(name, {}).update(values)

    predictions = columns.get("pred")
    for indices, names in masks.items():
        selected = {name: [columns[name][index] for index in indices] for name in names}
        complete_predictions = predictions is not None and all(predictions[index] is not None for index in indices)
        if complete_predictions:
            selected["pred"] = [predictions[index] for index in indices]
        metrics = process_validation_metrics(
            [sources[index] for index in indices], [uids[index] for index in indices], selected
        )
        merge(metrics, names)

        if predictions is not None and not complete_predictions:
            incomplete_groups = {
                (sources[index], uids[index]) for index in indices if predictions[index] is None
            }
            voting_indices = [index for index in indices if (sources[index], uids[index]) not in incomplete_groups]
            if voting_indices:
                voting = {name: [columns[name][index] for index in voting_indices] for name in names}
                voting["pred"] = [predictions[index] for index in voting_indices]
                metrics = process_validation_metrics(
                    [sources[index] for index in voting_indices],
                    [uids[index] for index in voting_indices],
                    voting,
                )
                merge(metrics, names, majority_only=True)
    return result
