__version__ = "0.0.3"


import sys


print(
    "\033[93mWARNING: Note that we are using the transformers and peft packages from the source directory, "
    "not the installed package. BitsandBytes 4bit quantizaztion was only working with the "
    "main brach of transformers and peft. Once transformers version 4.31.0 and peft version 0.4.0 is "
    "published to pypi we will use the published version.\033[0m"
)
sys.path.append("llm_toys/hf_src/transformers/src")
sys.path.append("llm_toys/hf_src/peft/src")
