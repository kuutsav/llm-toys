from llm_toys.config import EOC_FORMAT


#  Paraphrasing + Tone change

PARAPHRASE_PREDICT_FORMAT = (
    "### Instruction:\nGenerate a paraphrase for the following Input sentence.\n\n"
    "### Input:\n{input_text}\n\n### Response:\n"
)
PARAPHRASE_TRAIN_FORMAT = PARAPHRASE_PREDICT_FORMAT + "{response}\n\n" + EOC_FORMAT

TONE_CHANGE_PREDICT_FORMAT = (
    "### Instruction:\nChange the tone of the following Input sentence to {tone}.\n\n"
    "### Input:\n{input_text}\n\n### Response:\n"
)
TONE_CHANGE_TRAIN_FORMAT = TONE_CHANGE_PREDICT_FORMAT + "{response}\n\n" + EOC_FORMAT
