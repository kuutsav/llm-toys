# llm-toys

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![py\_versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/llm-toys/)

Small(7B and below), production-ready finetuned LLMs for a diverse set of useful tasks.

Supported tasks: Paraphrasing, Changing the tone of a passage, Summary and Topic generation from a dailogue,
~~Retrieval augmented QA(WIP)~~.

We finetune LoRAs on quantized 3B and 7B models. The 3B model is finetuned on specific tasks, while the 7B model is
finetuned on all the tasks.

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
# "We offer a satisfaction guarantee for our products."
```

#### Tone change

```python
paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", tone="casual")
# "We guarantee satisfaction with our products."

paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", tone="professional")
# "Our products are backed by a satisfaction guarantee."

paraphraser.paraphrase("Our prduucts come iwth a satisfaction guarantee.", tone="witty")
# "When you buy from us, you get a satisfaction guarantee! We're not joking around!"
```

#### Dialogue Summary and Topic generation

```python
from llm_toys.tasks import SummaryAndTopicGenerator

summary_theme_generator = SummaryAndTopicGenerator()
summary_theme_generator.generate_summary_and_topic(
    """
    #Person1#: I'm so excited for the premiere of the latest Studio Ghibli movie!
    #Person2#: What's got you so hyped?
    #Person1#: Studio Ghibli movies are pure magic! The animation, storytelling, everything is incredible.
    #Person2#: Which movie is it?
    #Person1#: It's called "Whisper of the Wind." It's about a girl on a magical journey to save her village.
    #Person2#: Sounds amazing! I'm in for the premiere.
    #Person1#: Great! We're in for a visual masterpiece and a heartfelt story.
    #Person2#: Can't wait to be transported to their world.
    #Person1#: It'll be an unforgettable experience, for sure!
    """.strip()
)
# {"summary": "#Person1# is excited for the premiere of the latest Studio Ghibli movie.
#              #Person1# thinks the animation, storytelling, and heartfelt story will be unforgettable.
#              #Person2# is also excited for the premiere.",
#  "topic": "Studio ghibli movie"}
```

## Training

### Data

- [Paraphrasing and Tone change](data/paraphrase_tone.json): Contains passages and their paraphrased versions as well
as the passage in different tones like casual, professional and witty. Used to models to rephrase and change the
tone of a passage. Data was generated using gpt-35-turbo. A small sample of training passages have also been picked
up from quora quesions and squad_2 datasets.

- [Dialogue Summary and Topic generation](data/dialogue_summary_topic.json): Contains Dialogues and their Summary
and Topic. The training data is ~1k records from the training split of the
[Dialogsum dataset](https://github.com/cylnlp/dialogsum). It also contains ~20 samples from the dev split.
Data points with longer Summaries and Topics were given priority in the sampling. Note that some(~30) topics
were edited manually in final training data as the original labeled Topic was just a word and not descriptive enough.

## Evaluation

### Paraphrasing and Tone change

WIP

### Dialogue Summary and Topic generation

Evaluation is done on 500 records from the [Dialogsum test](https://github.com/cylnlp/dialogsum/tree/main/DialogSum_Data)
split.

```python
{"rouge1": 0.453, "rouge2": 0.197, "rougeL": 0.365, "topic_similarity": 0.888}

{'rouge1': 0.448, 'rouge2': 0.195, 'rougeL': 0.359, 'topic_similarity': 0.886}
```

## Roadmap

- [ ] Add tests.
- [ ] Ability to switch the LoRAs(for task wise models) without re-initializing the backbone model and tokenizer.
- [ ] Retrieval augmented QA.
- [ ] Explore the generalizability of 3B model across more tasks.
- [ ] Explore even smaller models.
- [ ] Evaluation strategy for tasks where we don"t have a test/eval dataset handy.
