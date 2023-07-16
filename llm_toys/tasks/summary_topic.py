from time import time

from llm_toys.config import MODEL_CONFIG, SUPPORTED_END_TONES, SUPPORTED_MODEL_SIZES
from llm_toys.prompts import DIALOGUE_SUMMARY_TOPIC_PREDICT_FORMAT
from llm_toys.tasks.predict import PeftQLoraPredictor
from llm_toys.utils import print_warning


TXTS_TO_EXPLORE = [
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
]


class SummaryAndTopicGenerator(PeftQLoraPredictor):
    dialogue_summary_topic_format = DIALOGUE_SUMMARY_TOPIC_PREDICT_FORMAT

    def __init__(self, model_size: str = "3B", max_length: int = 512):
        assert (
            model_size.upper() in SUPPORTED_MODEL_SIZES
        ), f"Invalid model size; Supported sizes: {SUPPORTED_MODEL_SIZES}"
        super().__init__(**MODEL_CONFIG[model_size.upper()]["dialogue_summary_topic"], max_length=max_length)

    def generate_summary_and_topic(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        adjust_token_lengths: bool = False,
    ) -> dict[str, str]:
        input_text = self.dialogue_summary_topic_format.format(input_text=input_text)
        try:
            pred = super().predict(
                input_text, max_new_tokens, temperature, top_p, adjust_token_lengths=adjust_token_lengths
            )[0]
        except IndexError:
            print_warning(
                "Failed prediction; This could be due to a long input or a small value for `max_new_tokens`. "
                "If your input is long, try with `adjust_token_lengths=True`. Try increasing `max_new_tokens` otherwise."
            )
            return {"summary": "", "topic": ""}
        topic_ix = pred.index("# Topic:")
        summary = pred[len("# Summary:") : topic_ix].strip()
        topic = pred[topic_ix + len("# Topic:") :].strip()
        return {"summary": summary, "topic": topic}

    def explore(self, input_texts: list[str] = None, temperatures: list[float] = [0.1, 0.3, 0.5, 0.8, 1.0]) -> None:
        """This method can be used to explore outputs on a range of temperatures.
        This can help in finding the best temperature that suits the user."""
        preds, total_time = [], 0
        for input_text in input_texts or TXTS_TO_EXPLORE:
            print(f"\n{input_text}\n{'-'*110}")
            for i, temperature in enumerate(temperatures):
                print(f"Temperature: {temperature}")
                st = time()
                pred = self.generate_summary_and_topic(input_text, temperature=temperature)
                total_time += time() - st
                print(f" ↪ summary: {pred['summary']}")
                print(f" ↪ topic: {pred['topic']}")
                preds.append(pred)

                if i != len(temperatures):
                    print("\n")

        # This is rough count of the tokens as the model also generates new lines and EOC tokens.
        n_tokens = sum(
            (len(self.tokenizer.encode(p["summary"])) + len(self.tokenizer.encode(p["topic"])) + 20 for p in preds)
        )
        n_tokens_per_sec = n_tokens / total_time
        print(f"---\nGenerated {len(preds)} predictions in {total_time:.2f}s: {n_tokens_per_sec:.2f} tokens/s")
