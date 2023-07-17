from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig


# General

DEVICE = "mps" if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).parent.parent / "data"


# Model and Prompt related


class TaskType(str, Enum):
    PARAPHRASE_TONE = "paraphrase_tone"
    DIALOGUE_SUMMARY_TOPIC = "dialogue_summary_topic"


RESPONSE_FORMAT = "\n\n### Response:"
EOC_FORMAT = "\n\n### END"
SUPPORTED_MODEL_SIZES = ["3B", "7B"]
DEFAULT_3B_MODEL = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
DEFAULT_7B_MODEL = "tiiuae/falcon-7b"
MODEL_CONFIG = {
    "3B": {
        TaskType.PARAPHRASE_TONE: {
            "model_name": DEFAULT_3B_MODEL,
            "peft_model_id": "llm-toys/RedPajama-INCITE-Base-3B-v1-paraphrase-tone",
        },
        TaskType.DIALOGUE_SUMMARY_TOPIC: {
            "model_name": DEFAULT_3B_MODEL,
            "peft_model_id": "llm-toys/RedPajama-INCITE-Base-3B-v1-dialogue-summary-topic",
        },
    },
    "7B": {
        TaskType.PARAPHRASE_TONE: {
            "model_name": DEFAULT_7B_MODEL,
            "peft_model_id": "models/tiiuae/falcon-7b-paraphrase-tone-dialogue-summary-topic",
        },
        TaskType.DIALOGUE_SUMMARY_TOPIC: {
            "model_name": DEFAULT_7B_MODEL,
            "peft_model_id": "models/tiiuae/falcon-7b-paraphrase-tone-dialogue-summary-topic",
        },
    },
}
SUPPORTED_END_TONES = ["casual", "professional", "witty"]
TASK_TYPES = [tt.value for tt in TaskType]


def get_bnb_config() -> BitsAndBytesConfig:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
