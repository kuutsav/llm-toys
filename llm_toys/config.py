from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig


DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_3B_MODEL = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
TASK_TYPES = {"paraphrase_tone"}


def get_bnb_config() -> BitsAndBytesConfig:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
