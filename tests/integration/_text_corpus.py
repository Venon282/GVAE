"""Synthetic grammar corpus used for the text measurements."""

import random

import torch

PAD = 0
LEXICON = {
    "det": list(range(1, 3)),  # 2
    "adj": list(range(3, 11)),  # 8
    "noun": list(range(11, 21)),  # 10
    "verb": list(range(21, 29)),  # 8
    "adv": list(range(29, 32)),  # 3
}
TEMPLATES = [
    ("det", "adj", "noun", "verb", "det", "noun"),
    ("det", "noun", "verb", "det", "adj", "noun"),
    ("det", "adj", "noun", "verb", "adv"),
    ("det", "noun", "verb"),
]
VOCAB = 32
LENGTH = 8


def makeIds(n: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        template = rng.choice(TEMPLATES)
        ids = [rng.choice(LEXICON[slot]) for slot in template]
        rows.append(ids + [PAD] * (LENGTH - len(ids)))
    return torch.tensor(rows, dtype=torch.long)
