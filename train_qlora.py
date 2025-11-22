# train_qlora.py
\"\"\"TinyLlama-1.1B-Chat-v1.0 に対する QLoRA（4bit量子化 + LoRA）サンプル。
QLoRA アダプタは models/ft_qlora 以下に保存されます。
\"\"\"import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from training_utils import (
    TrainConfig,
    load_base_tokenizer,
    make_sft_dataset,
    build_trainer,
    ensure_dir,
)


def main():
    cfg = TrainConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_path="data/toy_qa.jsonl",
        output_dir="models/ft_qlora",
        num_train_epochs=1.0,
        learning_rate=2e-4,
    )

    ensure_dir(cfg.output_dir)

    tokenizer = load_base_tokenizer(cfg.model_name_or_path)

    # 4bit 量子化設定（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    dataset = make_sft_dataset(
        tokenizer=tokenizer,
        data_path=cfg.data_path,
        max_length=cfg.max_length,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=dataset,
        cfg=cfg,
        run_name="tinyllama_qlora_ft",
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved TinyLlama QLoRA adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
