from time import time

from llm_toys.config import MODEL_CONFIG, SUPPORTED_MODEL_SIZES
from llm_toys.prompts import PARAPHRASE_PREDICT_FORMAT
from llm_toys.tasks.predict import PeftQLoraPredictor


TXTS_TO_EXPLORE = """
"Thans for reaching out. Let me asist you with your isue."
"I undrstand you're facing an isue. Lts find a solution together."
"Sory for the inconvinience. I'll do my est to hlp you out."
"I'm srry, but I'm unble to resolv this at the moment. Plase try again later."
"Thnak you for brining this to my atention. I'll look into it right away."
"It appeas that you're facing a technial difficulty. Let's wk through it."
"I appoligize for the inconveience cause by our system. We're workng on a fix."
"Plase provice me with more dtails so I can bettr assist you."
"I unerstand that you're frustated. Rest asured, I'm her to help."
"Than you for connting us. We value your feedbak and will improve our servce."
""".strip()


class ParaPhraser(PeftQLoraPredictor):
    prompt = PARAPHRASE_PREDICT_FORMAT

    def __init__(self, model_size: str = "3B", max_length: int = 512):
        assert (
            model_size.upper() in SUPPORTED_MODEL_SIZES
        ), f"Invalid model size; Supported sizes: {SUPPORTED_MODEL_SIZES}"
        super().__init__(**MODEL_CONFIG[model_size.upper()], max_length=max_length)

    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        adjust_token_lengths: bool = False,
    ):
        input_text = self.prompt.format(input_text=input_text)
        return super().predict(
            input_text,
            max_new_tokens,
            temperature,
            top_p,
            num_return_sequences,
            adjust_token_lengths,
        )

    def explore(self, input_texts: list[str] = None, temperatures: list[float] = [0.1, 0.3, 0.5, 0.8, 1.0]):
        preds, total_time = [], 0
        for input_text in input_texts or TXTS_TO_EXPLORE.split("\n"):
            print(f"\n{input_text}\n{'-'*len(input_text)}")
            for temperature in temperatures:
                st = time()
                pred = self.predict(input_text, temperature=temperature)[0]
                total_time += time() - st
                print(f"Temperature: {temperature} -> {pred}")
                preds.append(pred)

        n_tokens = sum((len(self.tokenizer.encode(txt)) for txt in preds))
        n_tokens_per_sec = n_tokens / total_time
        print(f"---\nGenerated {len(preds)} predictions in {total_time:.2f}s: {n_tokens_per_sec:.2f} tokens/s")
