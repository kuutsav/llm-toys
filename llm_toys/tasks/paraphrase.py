from time import time

from llm_toys.config import MODEL_CONFIG, SUPPORTED_END_TONES, SUPPORTED_MODEL_SIZES
from llm_toys.prompts import PARAPHRASE_PREDICT_FORMAT, TONE_CHANGE_PREDICT_FORMAT
from llm_toys.tasks.predict import PeftQLoraPredictor
from llm_toys.utils import print_warning


TXTS_TO_EXPLORE = """
"I undrstand you're facing an isue. Lts find a solution together."
"I'm srry, but I'm unble to resolv this at the moment. Plase try again later."
"It appeas that you're facing a technial difficulty. Let's wk through it."
"I appoligize for the inconveience cause by our system. We're workng on a fix."
"Plase provice me with more dtails so I can bettr assist you."
""".strip()


class Paraphraser(PeftQLoraPredictor):
    paraphrase_format = PARAPHRASE_PREDICT_FORMAT
    tone_change_format = TONE_CHANGE_PREDICT_FORMAT

    def __init__(self, model_size: str = "3B", max_length: int = 512):
        assert (
            model_size.upper() in SUPPORTED_MODEL_SIZES
        ), f"Invalid model size; Supported sizes: {SUPPORTED_MODEL_SIZES}"
        super().__init__(**MODEL_CONFIG[model_size.upper()], max_length=max_length)

    def paraphrase(
        self,
        input_text: str,
        tone: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        adjust_token_lengths: bool = False,
    ):
        if tone:
            if tone.lower() not in SUPPORTED_END_TONES:
                print_warning(
                    f"Model has been fine-tuned to change the tone to one of following: {SUPPORTED_END_TONES}; "
                    "Using a different tone may result in unexpected output."
                )
            input_text = self.tone_change_format.format(input_text=input_text, tone=tone)
        else:
            input_text = self.paraphrase_format.format(input_text=input_text)
        return super().predict(
            input_text,
            max_new_tokens,
            temperature,
            top_p,
            num_return_sequences,
            adjust_token_lengths,
        )

    def explore(self, input_texts: list[str] = None, temperatures: list[float] = [0.1, 0.3, 0.5, 0.8, 1.0]):
        """This method can be used to explore outputs on a range of temperatures.
        This can help in finding the best temperature that suits the user.
        """
        preds, total_time = [], 0
        for input_text in input_texts or TXTS_TO_EXPLORE.split("\n"):
            print(f"\n{input_text}\n{'-'*len(input_text)}")
            for i, temperature in enumerate(temperatures):
                print(f"Temperature: {temperature}")
                for tone in [None] + SUPPORTED_END_TONES:
                    st = time()
                    pred = self.paraphrase(input_text, tone=tone, temperature=temperature)[0]
                    total_time += time() - st
                    print(f" ↪ {tone or 'original'}: {pred}")
                    preds.append(pred)
                if i != len(temperatures):
                    print("\n")

        n_tokens = sum((len(self.tokenizer.encode(txt)) for txt in preds))
        n_tokens_per_sec = n_tokens / total_time
        print(f"\n---\nGenerated {len(preds)} predictions in {total_time:.2f}s: {n_tokens_per_sec:.2f} tokens/s")
