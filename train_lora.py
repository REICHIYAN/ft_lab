"""TinyLlama-1.1B-Chat-v1.0 に LoRA を挿入して微調整するサンプル。
LoRA アダプタは models/ft_lora 以下に保存されます。
"""
from peft import LoraConfig, get_peft_model
from training_utils import (
    TrainConfig,
    load_base_tokenizer,
    load_base_model,
    make_sft_dataset,
    build_trainer,
    ensure_dir,
)


def main():
    cfg = TrainConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_path="data/toy_qa.jsonl",
        output_dir="models/ft_lora",
        num_train_epochs=1.0,
        learning_rate=2e-4,
    )

    ensure_dir(cfg.output_dir)

    tokenizer = load_base_tokenizer(cfg.model_name_or_path)
    base_model = load_base_model(cfg.model_name_or_path)

    # Llama 系 TinyLlama 向けの LoRA 挿入先
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
        run_name="tinyllama_lora_ft",
    )

    trainer.train()
    # LoRA アダプタを保存
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved TinyLlama LoRA adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
