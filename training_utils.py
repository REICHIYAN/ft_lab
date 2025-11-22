# training_utils.py
import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# TinyLlama をデフォルトモデルに設定
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass
class TrainConfig:
    model_name_or_path: str = DEFAULT_MODEL_NAME
    data_path: str = "data/toy_qa.jsonl"
    output_dir: str = "models/ft_full"
    max_length: int = 512
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    learning_rate: float = 5e-5
    logging_steps: int = 10
    save_steps: int = 50
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1


def load_base_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TinyLlama は Llama 系なので、pad_token を EOS に揃えて右詰めにする
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer の語彙サイズに合わせて埋め込みをリサイズ
    model.resize_token_embeddings(len(model.get_input_embeddings().weight))
    return model


def make_sft_dataset(
    tokenizer: AutoTokenizer,
    data_path: str,
    max_length: int = 512,
):
    """
    JSONL: {"question": ..., "answer": ...}
    を Instruction-following っぽいフォーマットに変換してトークナイズします。
    """
    raw = load_dataset("json", data_files=data_path)

    def _format(example):
        prompt = f"User: {example['question']}\nAssistant:"
        text = prompt + " " + example["answer"]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # LM学習用に labels を input_ids と同じにする
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = raw["train"].map(
        _format,
        remove_columns=raw["train"].column_names,
    )
    return tokenized


def build_trainer(
    model,
    tokenizer,
    tokenized_dataset,
    cfg: TrainConfig,
    run_name: Optional[str] = None,
) -> Trainer:
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=1,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    return trainer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
