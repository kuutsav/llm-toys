from __future__ import annotations
import json
from typing import TYPE_CHECKING

from llm_toys.config import DATA_DIR
from llm_toys.prompts import PARAPHRASE_TRAIN_FORMAT, TONE_CHANGE_TRAIN_FORMAT


if TYPE_CHECKING:
    from pathlib import Path

paraphrase_tone_data_f = DATA_DIR / "paraphrase_tone.json"


def load_json(fname: str | Path) -> dict:
    with open(fname, "r") as f:
        return json.load(f)


def save_json(fname: str | Path, data: dict) -> None:
    with open(fname, "w") as f:
        json.dump(data, f)


def save_text(fname: str | Path, data: str) -> None:
    with open(fname, "w") as f:
        f.write(data)


def token_stats(texts: list[str]) -> None:
    from transformers import AutoTokenizer
    from llm_toys.config import DEFAULT_3B_MODEL

    tokenier = AutoTokenizer.from_pretrained(DEFAULT_3B_MODEL)

    token_lengths = [len(tokenier.encode(txt)) for txt in texts]
    print(f"# Data points    : {len(texts)}")
    print(f"Min token len    : {min(token_lengths)}")
    print(f"Max token len    : {max(token_lengths)}")
    print(f"Mean token len   : {int(sum(token_lengths) / len(token_lengths))}")
    print(f"Median token len : {sorted(token_lengths)[len(token_lengths) // 2]}")
    print(f"Q3 token len     : {sorted(token_lengths)[int(len(token_lengths) * 0.75)]}")


def paraphrase_tone_training_data(print_token_stats: bool = False) -> list[str]:
    """Generates training data for both `Paraphrasing` and `Tone Change`. Also prints out the
    token stats for the training data to help with picking the `max_length` during training."""
    data = load_json(paraphrase_tone_data_f)

    training_data = []
    for d in data:
        # Tone: Original -> Casual, Professional, Witty
        # Tone: Casual -> Professional, Witty
        for start_tone in ["original"]:
            for end_tone in ["casual", "professional", "witty"]:
                if start_tone == end_tone:
                    continue
                training_data.append(
                    TONE_CHANGE_TRAIN_FORMAT.format(tone=end_tone, input_text=d[start_tone], response=d[end_tone])
                )
        # Paraphrase: Original -> Paraphrase
        training_data.append(PARAPHRASE_TRAIN_FORMAT.format(input_text=d["original"], response=d["paraphrase"]))

    if print_token_stats:
        token_stats(training_data)

    return training_data
