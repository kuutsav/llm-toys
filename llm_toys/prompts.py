#  Paraphrasing + Tone change

PARAPHRASE_INSTRUCTION = (
    "### Instruction:\nGenerate a paraphrase for the following Input sentence.\n\n"
    "### Input:\n{input_text}\n\n### Response:\n{response}"
)
TONE_CHANGE_INSTRUCTION = (
    "### Instruction:\nChange the tone of the following Input sentence to {tone}.\n\n"
    "### Input:\n{input_text}\n\n### Response:\n{response}"
)
