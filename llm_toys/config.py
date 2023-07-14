from pathlib import Path

import torch
from transformers import BitsAndBytesConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).parent.parent / "data"

EOC_FORMAT = "### END"
DEFAULT_3B_MODEL = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
DEFAULT_7B_MODEL = "tiiuae/falcon-7b"

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
