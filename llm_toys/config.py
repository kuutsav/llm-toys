from pathlib import Path

import torch
from transformers import BitsAndBytesConfig


DEVICE = "mps" if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).parent.parent / "data"

EOC_FORMAT = "### END"
SUPPORTED_MODEL_SIZES = {"3B", "7B"}
DEFAULT_3B_MODEL = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
DEFAULT_7B_MODEL = "tiiuae/falcon-7b"
MODEL_CONFIG = {
    "3B": {
        "model_name": DEFAULT_3B_MODEL,
        "peft_model_id": "llm-toys/RedPajama-INCITE-Base-3B-v1-paraphrase-tone",
    },
    "7B": {
        "model_name": DEFAULT_7B_MODEL,
        "peft_model_id": "llm-toys/falcon-7b-paraphrase-tone",
    },
}

SUPPORTED_END_TONES = ["casual", "professional", "witty"]
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
