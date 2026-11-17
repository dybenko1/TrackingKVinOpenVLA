"""Epoch-based batches with at most one caption per image in each batch."""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence


class CaptionRoundBatchSampler:
    """Use every annotation once, without same-image collisions within a batch.

    Each epoch independently shuffles each image's captions, consumes one
    caption per image per round, shuffles each round, and slices the round
    into batches. ``set_epoch`` makes ordering reproducible and epoch-specific.
    """

    def __init__(self, image_ids: Sequence[int], batch_size: int = 128, seed: int = 123):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.image_ids = tuple(int(value) for value in image_ids)
        if not self.image_ids:
            raise ValueError("At least one annotation is required")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._groups: dict[int, list[int]] = defaultdict(list)
        for row, image_id in enumerate(self.image_ids):
            self._groups[image_id].append(row)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        rng = random.Random((self.seed << 32) ^ self.epoch)
        shuffled = {}
        for image_id, rows in self._groups.items():
            rows = list(rows)
            rng.shuffle(rows)
            shuffled[image_id] = rows

        rounds: list[list[int]] = []
        max_captions = max(map(len, shuffled.values()))
        for round_index in range(max_captions):
            round_rows = [rows[round_index] for rows in shuffled.values() if round_index < len(rows)]
            rng.shuffle(round_rows)
            rounds.append(round_rows)

        batches = [round_rows[offset:offset + self.batch_size]
                   for round_rows in rounds
                   for offset in range(0, len(round_rows), self.batch_size)]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return sum(
            (sum(len(rows) > index for rows in self._groups.values()) + self.batch_size - 1)
            // self.batch_size
            for index in range(max(map(len, self._groups.values())))
        )
