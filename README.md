# llm-toys

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![py\_versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/llm-toys/)

Small(7B and below), production-ready finetuned LLMs for a diverse set of useful tasks.

Supported tasks: Paraphrasing, Changing the tone of a passage, Summarization and Theme detection from a conversation,
Retrieval augmented QA.

We finetune LoRAs on quantized 3B and 7B LLMS. The 3B model is finetuned on specific tasks, while the 7B model is
finetuned on all tasks.

The goal is to be able to finetune and use all these models on a very modest consumer grade hardware.

## Installation

```bash
pip install llm-toys
```

> Note that we are using the transformers and peft packages from the source directory, 
> not the installed package. 4bit bitsandbytes quantization was only working with the 
> main brach of transformers and peft. Once transformers version 4.31.0 and peft version 0.4.0 is 
> published to pypi we will use the published version.

## Usage

### Task specific 3B models
#### Paraphrasing
```python
from llm_toys.tasks import Paraphraser

paraphraser = Paraphraser()
paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.")
# "Our products are backed by a guarantee of your satisfaction."
```

#### Tone change
```python
paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", "formal", tone="casual")
# "We've got your back! Our products come with a satisfaction guarantee."

paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", "formal", tone="professional")
# "We stand behind our products with a satisfaction guarantee."

paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", "formal", tone="witty")
# "We put our money where our product is! Our satisfaction guarantee ensures you'll be doing happy
# dances with our products!"
```

## Roadmap

- [ ] Explore even smaller models.
- [ ] Explore the generalizability of 3B model across more tasks.
- [ ] Better evaluation datastes from a diverse set of domains.
