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

> Might not work without a CUDA enabled GPU
>
> If you encounter "The installed version of bitsandbytes was compiled without GPU support" with bitsandbytes
> then look here https://github.com/TimDettmers/bitsandbytes/issues/112
>
> or try
>
> cp <path_to_your_venv>/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so <path_to_your_venv>/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so 
>
> Note that we are using the transformers and peft packages from the source directory, 
> not the installed package. 4bit bitsandbytes quantization was only working with the 
> main brach of transformers and peft. Once transformers version 4.31.0 and peft version 0.4.0 is 
> published to pypi we will use the published version.

## Available Models

| Model | Size | Tasks | Colab |
| ----- | ---- | ----- | ----- |
| [llm-toys/RedPajama-INCITE-Base-3B-v1-paraphrase-tone](https://huggingface.co/llm-toys/RedPajama-INCITE-Base-3B-v1-paraphrase-tone) | 3B | Paraphrasing, Tone change | [Notebook](https://colab.research.google.com/drive/1MSl8IDLjs3rgEv8cPHbJLR8GHh2ucT3_) |
| [llm-toys/RedPajama-INCITE-Base-3B-v1-dialogue-summary-topic](https://huggingface.co/llm-toys/RedPajama-INCITE-Base-3B-v1-dialogue-summary-topic) | 3B | Dialogue Summary and Topic generation | [Notebook](https://colab.research.google.com/drive/1MSl8IDLjs3rgEv8cPHbJLR8GHh2ucT3_) |
| [llm-toys/falcon-7b-paraphrase-tone-dialogue-summary-topic](https://huggingface.co/llm-toys/falcon-7b-paraphrase-tone-dialogue-summary-topic) | 7B | Paraphrasing, Tone change, Dialogue Summary and Topic generation | [Notebook](https://colab.research.google.com/drive/1hhANNzQkxhrPIIrxtvf0WT_Ste8KrFjh#scrollTo=d6-OJJq_q5Qr) |

## Usage

### Task specific 3B models

#### Paraphrasing

```python
from llm_toys.tasks import Paraphraser

paraphraser = Paraphraser()
paraphraser.paraphrase("Hey, can yuo hepl me cancel my last order?")
# "Could you kindly assist me in canceling my previous order?"
```

#### Tone change

```python
paraphraser.paraphrase("Hey, can yuo hepl me cancel my last order?", tone="casual")
# "Hey, could you help me cancel my order?"

paraphraser.paraphrase("Hey, can yuo hepl me cancel my last order?", tone="professional")
# "I would appreciate guidance on canceling my previous order."

paraphraser.paraphrase("Hey, can yuo hepl me cancel my last order?", tone="witty")
# "Hey, I need your help with my last order. Can you wave your magic wand and make it disappear?"
```

#### Dialogue Summary and Topic generation

```python
from llm_toys.tasks import SummaryAndTopicGenerator

summary_topic_generator = SummaryAndTopicGenerator()
summary_topic_generator.generate_summary_and_topic(
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

### General 7B model

```python
from llm_toys.tasks import GeneralTaskAssitant
from llm_toys.config import TaskType

gta = GeneralTaskAssitant()
gta.complete(TaskType.PARAPHRASE_TONE, "Hey, can yuo hepl me cancel my last order?")
# "Could you assist me in canceling my previous order?"

gta.complete(TaskType.PARAPHRASE_TONE, "Hey, can yuo hepl me cancel my last order?", tone="casual")
# "Hey, can you help me cancel my last order?"

gta.complete(TaskType.PARAPHRASE_TONE, "Hey, can yuo hepl me cancel my last order?", tone="professional")
# "I would appreciate if you could assist me in canceling my previous order."

gta.complete(TaskType.PARAPHRASE_TONE, "Hey, can yuo hepl me cancel my last order?", tone="witty")
# "Oops! Looks like I got a little carried away with my shopping spree. Can you help me cancel my last order?"

chat = """
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
gta.complete(TaskType.DIALOGUE_SUMMARY_TOPIC, chat)
# {"summary": "#Person1# tells #Person2# about the upcoming Studio Ghibli movie.
#              #Person1# thinks it's magical and #Person2#'s excited to watch it.",
#  "topic": "Movie premiere"}
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

### Sample training script

To look at all the options

```bash
python llm_toys/train.py --help
```

To train a paraphrasing and tone change model

```bash
python llm_toys/train.py \
    --task_type paraphrase_tone \
    --model_name meta-llama/Llama-2-7b \
    --max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --eval_ratio 0.05
```

## Evaluation

### Paraphrasing and Tone change

WIP

### Dialogue Summary and Topic generation

Evaluation is done on 500 records from the [Dialogsum test](https://github.com/cylnlp/dialogsum/tree/main/DialogSum_Data)
split.

```python
# llm-toys/RedPajama-INCITE-Base-3B-v1-dialogue-summary-topic
{"rouge1": 0.453, "rouge2": 0.197, "rougeL": 0.365, "topic_similarity": 0.888}

# llm-toys/falcon-7b-paraphrase-tone-dialogue-summary-topic
{'rouge1': 0.448, 'rouge2': 0.195, 'rougeL': 0.359, 'topic_similarity': 0.886}
```

## Roadmap

- [ ] Add tests.
- [ ] Ability to switch the LoRAs(for task wise models) without re-initializing the backbone model and tokenizer.
- [ ] Retrieval augmented QA.
- [ ] Explore the generalizability of 3B model across more tasks.
- [ ] Explore even smaller models.
- [ ] Evaluation strategy for tasks where we don"t have a test/eval dataset handy.
- [ ] Data collection strategy and finetuning a model for OpenAI like "function calling"
