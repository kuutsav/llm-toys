from __future__ import annotations
from typing import TYPE_CHECKING

import torch.nn.functional as F

from llm_toys.config import DATA_DIR, DEVICE
from llm_toys.utils import load_json, save_json


if TYPE_CHECKING:
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel

dialogue_summary_topic_test_data_f = DATA_DIR / "dialogue_summary_topic_test.json"


def summary_topic_test_pred(model_size: str = "3B") -> None:
    """Generates summary and topic for each dialogue in the
    dialogue_summary_topic_test data and saves it to the same file."""
    from llm_toys.tasks import SummaryAndTopicGenerator

    model = SummaryAndTopicGenerator(model_size=model_size)
    test_data = load_json(dialogue_summary_topic_test_data_f)

    for i, d in enumerate(test_data, start=1):
        pred = model.generate_summary_and_topic(d["dialogue"])
        d["pred_summary"] = pred["summary"]
        d["pred_topic"] = pred["topic"]
        if i % 10 == 0:
            print(f"Processed {i:<3}/{len(test_data)} dialogues")

    save_json(dialogue_summary_topic_test_data_f, test_data)


def _max_rouge_score(scores: list) -> dict:
    return sorted(scores, key=lambda x: x["rouge1"].fmeasure, reverse=True)[0]


def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _encode(encoder: AutoModel, tokenizer: AutoTokenizer, input_texts: list[str]) -> Tensor:
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    batch_dict["input_ids"] = batch_dict["input_ids"].to(DEVICE)
    batch_dict["attention_mask"] = batch_dict["attention_mask"].to(DEVICE)
    outputs = encoder(input_ids=batch_dict["input_ids"], attention_mask=batch_dict["attention_mask"])
    embeddings = _average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def summary_topic_test_eval() -> dict[str, float]:
    """Generates evaluation metrics on dialogue_summary_topic_test after the predictions have been generated
    using summary_topic_test_pred().

    The eval metrics are:
    - rouge1, rouge2, rougeL scores for the predicted summary; max from the 3 ground truth summaries
    - topic similarity score between the predicted topic and the ground truth topics; max from the 3 ground truth topics
    """
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    encoder = AutoModel.from_pretrained("intfloat/e5-base-v2")
    encoder = encoder.to(DEVICE)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    test_data = load_json(dialogue_summary_topic_test_data_f)

    all_scores = []
    for d in test_data:
        if "pred_summary" in d and "pred_topic" in d:
            summary_scores = [scorer.score(d["pred_summary"], d[f"summary{i+1}"]) for i in range(2)]
            topic_scores = [_encode(encoder, tokenizer, [d["pred_topic"], d[f"topic{i+1}"]]) for i in range(2)]
            max_topic_sim_score = max([(a @ b.T).item() for a, b in topic_scores])
            scores = {**_max_rouge_score(summary_scores), "topic_similarity": max_topic_sim_score}
            all_scores.append(scores)

    avg_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "topic_similarity": 0.0}
    for s in all_scores:
        for k in avg_scores:
            if k == "topic_similarity":
                avg_scores[k] += s[k]
            else:
                avg_scores[k] += s[k].fmeasure
    for k in avg_scores:
        avg_scores[k] = round(avg_scores[k] / len(all_scores), 3)

    return avg_scores
