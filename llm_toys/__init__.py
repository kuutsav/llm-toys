__version__ = "0.1.1"


from pathlib import Path
import sys

from llm_toys.utils import print_warning


print_warning(
    "Note that we are using the transformers and peft packages from the source directory, "
    "not the installed package. 4bit bitsandbytes quantization was only working with the "
    "main brach of transformers and peft. Once transformers version 4.31.0 and peft version 0.4.0 is "
    "published to pypi we will use the published version."
)

sys.path.append(str(Path(__file__).parent / "hf"))
