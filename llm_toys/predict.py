from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from llm_toys.config import DEVICE, EOC_FORMAT, get_bnb_config


class StoppingCriteriaSub(StoppingCriteria):
    """Helps in stopping the generation when a certain sequence of tokens is generated."""

    def __init__(self, stops: list = []):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        return input_ids[0][-len(self.stops) :].tolist() == self.stops


class PeftQLoraPredictor:
    """Peft quantized lora model predictor for a model finetuned on generative task."""

    def __init__(self, model_name: str, peft_model_id: str, max_length: int = 512, skip_model_int: bool = False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self._eoc_ids = self.tokenizer(EOC_FORMAT, return_tensors="pt")["input_ids"][0]

        if not skip_model_int:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": 0},
                trust_remote_code=True,
                quantization_config=get_bnb_config(),
            )
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            self.model = self.model.to(DEVICE)
            self.model.eval()

        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=self.tokenizer(EOC_FORMAT)["input_ids"])]
        )

    def _adjust_token_lengths(self, tokenized: dict) -> None:
        """If tokenized length is more than max length, we truncate to
        include the RESPONSE ids to get a prediction from the model."""
        for i in range(len(tokenized["input_ids"])):
            if len(tokenized["input_ids"][i]) == self.max_length:
                tokenized["input_ids"][i] = torch.concat(
                    (tokenized["input_ids"][i][: -len(self._end_ids)], self._end_ids)
                )

    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        adjust_token_lengths: bool = False,
    ) -> list[str]:
        """If you know that the input token length is going to exceed the max length then set adjust_token_lengths
        to True so that the EOC tokens can be added back to the truncated tokens."""
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if adjust_token_lengths:
            self._adjust_token_lengths(tokenized)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=tokenized["input_ids"].to("cuda"),
                attention_mask=tokenized["attention_mask"].to("cuda"),
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                stopping_criteria=self.stopping_criteria,
            )

        out_texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]
        preds = []
        for o in out_texts:
            response_ix = o.find("### Response:")
            if response_ix != -1:
                response_txt = o.split("### Response:")[1]
                end_ix = response_txt.find(EOC_FORMAT)
                if end_ix != -1:
                    preds.append(response_txt[:end_ix].strip())
        return preds
