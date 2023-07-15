import argparse
import random

from loguru import logger
from torch.utils.data import Dataset

from llm_toys.config import DEFAULT_3B_MODEL, get_bnb_config, TASK_TYPES
from llm_toys.hf.peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from llm_toys.hf.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from llm_toys.utils import paraphrase_tone_training_data


class CustomDataset(Dataset):
    """Custom dataset for the trainer. Tokenizes and keeps the
    entire dataset in memory hence it's not suited for large datasets."""

    def __init__(self, tokenizer: AutoTokenizer, text: str, max_length: int = 1024):
        self.text = text
        self.tokenized = tokenizer(
            self.text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __getitem__(self, ix):
        return self.tokenized["input_ids"][ix]

    def __len__(self):
        return len(self.text)


class PeftQLoraTrainer:
    """Contains helpful methods for loading+transforming the data and training the model."""

    def __init__(self, args: dict):
        self.args = args
        assert (
            args["task_type"].lower() in TASK_TYPES
        ), f"Invalid task type {args['task_type']}; Valid task types are {TASK_TYPES}"
        self.experiment_name = "-".join(args["task_type"].split("_")).lower()
        self._models_initialized = False

        if self.args["use_aim"]:
            from aim import Run

            self.run = Run(experiment=self.experiment_name)

    def _set_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.args["model_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _set_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args["model_name"],
            device_map={"": 0},
            trust_remote_code=True,
            quantization_config=get_bnb_config(),
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            r=self.args["lora_r"],
            lora_alpha=self.args["lora_alpha"],
            target_modules=(None if self.args["model_name"].startswith("gpt2") else ["query_key_value"]),
            lora_dropout=self.args["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

    def load_data(self) -> list[str]:
        if self.args["task_type"].lower() == "paraphrase_tone":
            return paraphrase_tone_training_data()
        raise ValueError(f"Probably an invalid task type {self.args['task_type']}; Valid task types are {TASK_TYPES}")

    @staticmethod
    def get_train_evaluation_texts(eval_ratio: float, data: list[str]) -> tuple[list[str], list[str]]:
        val_ixs = set()
        if eval_ratio:
            n_val = int(eval_ratio * len(data))
            val_ixs = set(random.sample(range(len(data)), k=n_val))
        train_ixs = (i for i in range(len(data)) if i not in val_ixs)
        train_texts = [data[i] for i in train_ixs]
        val_texts = [data[i] for i in val_ixs]
        return train_texts, val_texts

    def set_trainer(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        training_args = TrainingArguments(
            auto_find_batch_size=True if self.args["batch_size"] else False,
            per_device_train_batch_size=self.args["batch_size"],
            per_device_eval_batch_size=self.args["batch_size"],
            gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
            num_train_epochs=self.args["num_train_epochs"],
            learning_rate=self.args["learning_rate"],
            fp16=True,
            do_eval=self.args["eval_ratio"],
            logging_strategy="steps",
            logging_steps=self.args["logging_steps"],
            evaluation_strategy="steps",
            eval_steps=self.args["eval_steps"],
            output_dir=f"models/{self.args['model_name']}",
            overwrite_output_dir=True,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def log_params(self):
        """Logs all the metadata and enables a UI to observe/compare them using Aim."""
        hparams = self.args
        self.run["hparams"] = hparams

        for log in self.trainer.state.log_history:
            if "eval_loss" in log:
                self.run.track(log["eval_loss"], name="loss", step=log["step"], context={"subset": "eval"})
            elif "loss" in log:
                self.run.track(log["loss"], name="loss", step=log["step"], context={"subset": "train"})

    def prep_data_train_log_and_save_model(self):
        """Loads the data, preps the train and eval datasets, trains+logs+saves the model."""
        if not self._models_initialized:
            self._set_tokenizer()
            self._set_model()
            self._models_initialized = True

        data = self.load_data()
        logger.info(f"Loaded training data; {len(data)} records")

        random.seed(self.args["seed"])
        train_texts, eval_texts = self.get_train_evaluation_texts(self.args["eval_ratio"], data)
        logger.info(f"{'# Train':<15} : {len(train_texts)}")
        if self.args["eval_ratio"]:
            logger.info(f"{'# Eval':<15} : {len(eval_texts)}")

        train_dataset = CustomDataset(self.tokenizer, train_texts, max_length=self.args["max_length"])
        logger.info(f"{'# Train dataset':<15} : {len(train_dataset)}")
        if self.args["eval_ratio"]:
            eval_dataset = CustomDataset(self.tokenizer, eval_texts, max_length=args.max_length)
            logger.info(f"{'# Eval dataset':<15} : {len(eval_dataset)}")

        self.set_trainer(train_dataset, eval_dataset if self.args["eval_ratio"] else None)
        self.model.config.use_cache = False
        self.trainer.train()

        self.model.save_pretrained(f"models/{self.args['model_name']}-{self.experiment_name}")

        if self.args["use_aim"]:
            self.log_params()

    def push_to_hub(self):
        self.model.push_to_hub(f"{self.args['model_name']}-{self.experiment_name}", private=True, use_auth_token=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, help=f"Valid task types are {TASK_TYPES}.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_3B_MODEL, help="Model name.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed.")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Ratio of evaluation data.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of input sequence.")
    parser.add_argument("--lora_r", type=int, default=16, help="Lora r.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Lora dropout.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="# Epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval steps.")
    parser.add_argument("--use_aim", action=argparse.BooleanOptionalAction, help="To use AIM for logging or not.")
    args = parser.parse_args()

    peft_qlora_trainer = PeftQLoraTrainer(vars(args))
    peft_qlora_trainer.prep_data_train_log_and_save_model()
