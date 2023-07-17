from time import time

from llm_toys.config import MODEL_CONFIG, SUPPORTED_END_TONES, SUPPORTED_MODEL_SIZES_ALL_TASKS, TASK_TYPES, TaskType
from llm_toys.prompts import (
    DIALOGUE_SUMMARY_TOPIC_PREDICT_FORMAT,
    PARAPHRASE_PREDICT_FORMAT,
    TONE_CHANGE_PREDICT_FORMAT,
)
from llm_toys.tasks.predict import PeftQLoraPredictor
from llm_toys.tasks.summary_topic import SummaryAndTopicGenerator
from llm_toys.utils import print_warning


TXTS_TO_EXPLORE = [
    (
        TaskType.PARAPHRASE_TONE,
        """
I undrstand you're facing an isue. Lts find a solution together.---
I'm srry, but I'm unble to resolv this at the moment. Plase try again later.
""".strip(),
    ),
    (
        TaskType.DIALOGUE_SUMMARY_TOPIC,
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
---
#Person1#: Have you seen the floods happening worldwide? Climate change is causing havoc.
#Person2#: Yes, it's distressing. We must act fast.
#Person1#: Rising temperatures intensify rainfall, leading to severe floods.
#Person2#: Vulnerable communities suffer the most. We need global cooperation.
#Person1#: Mitigation and adaptation strategies are crucial.
#Person2#: Governments and international organizations must act now.
#Person1#: Let's push for a sustainable future. Time is running out.
""".strip(),
    ),
]


class GeneralTaskAssitant(PeftQLoraPredictor):
    paraphrase_format = PARAPHRASE_PREDICT_FORMAT
    tone_change_format = TONE_CHANGE_PREDICT_FORMAT
    dialogue_summary_topic_format = DIALOGUE_SUMMARY_TOPIC_PREDICT_FORMAT

    def __init__(self, model_size: str = "7B", max_length: int = 512):
        assert (
            model_size.upper() in SUPPORTED_MODEL_SIZES_ALL_TASKS
        ), f"Invalid model size; Supported sizes: {SUPPORTED_MODEL_SIZES_ALL_TASKS}"
        super().__init__(**MODEL_CONFIG[model_size.upper()]["all"], max_length=max_length)

    def complete(
        self,
        task_type: TaskType,
        input_text: str,
        tone: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        adjust_token_lengths: bool = False,
    ) -> str | dict[str, str]:
        if task_type == TaskType.PARAPHRASE_TONE:
            if tone:
                if tone.lower() not in SUPPORTED_END_TONES:
                    print_warning(
                        f"Model has been fine-tuned to change the tone to one of following: {SUPPORTED_END_TONES}; "
                        "Using a different tone may result in unexpected output."
                    )
                input_text = self.tone_change_format.format(input_text=input_text, tone=tone)
            else:
                input_text = self.paraphrase_format.format(input_text=input_text)
        elif task_type == TaskType.DIALOGUE_SUMMARY_TOPIC:
            input_text = self.dialogue_summary_topic_format.format(input_text=input_text)
        else:
            raise ValueError(f"Invalid task type {task_type}; Valid task types are {TASK_TYPES}")
        pred = super().predict(
            input_text, max_new_tokens, temperature, top_p, adjust_token_lengths=adjust_token_lengths
        )[0]

        if task_type == TaskType.DIALOGUE_SUMMARY_TOPIC:
            return SummaryAndTopicGenerator.parse_summary_and_topic(pred)
        return pred

    def explore(
        self, input_task_texts: list[tuple[TaskType, str]] = None, temperatures: list[float] = [0.1, 0.5]
    ) -> None:
        """This method can be used to explore outputs on a range of temperatures.
        This can help in finding the best temperature that suits the user."""
        preds, total_time = [], 0
        for task_type, input_texts in input_task_texts or TXTS_TO_EXPLORE:
            for input_text in input_texts.split("---"):
                input_text = input_text.strip()
                print(f"\n{input_text}\n{'-'*min(110, len(input_text))}")
                for i, temperature in enumerate(temperatures):
                    if i != 0:
                        print()
                    print(f"Temperature: {temperature}")
                    if task_type == TaskType.PARAPHRASE_TONE:
                        for tone in [None] + SUPPORTED_END_TONES:
                            st = time()
                            pred = self.complete(task_type, input_text, tone=tone, temperature=temperature)
                            total_time += time() - st
                            print(f"  ↪ {tone or 'paraphrase'}: {pred}")
                            preds.append(pred)
                    elif task_type == TaskType.DIALOGUE_SUMMARY_TOPIC:
                        st = time()
                        pred = self.complete(task_type, input_text, temperature=temperature)
                        total_time += time() - st
                        print(f"  ↪ summary: {pred['summary']}")
                        print(f"  ↪ topic: {pred['topic']}")
                        preds.append(pred["summary"] + pred["topic"])

        n_tokens = sum((len(self.tokenizer.encode(txt)) for txt in preds))
        n_tokens_per_sec = n_tokens / total_time
        print(f"\n---\nGenerated {len(preds)} predictions in {total_time:.2f}s: {n_tokens_per_sec:.2f} tokens/s")
