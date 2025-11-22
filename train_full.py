# train_full.py
\"\"\"TinyLlama-1.1B-Chat-v1.0 を Full Fine-tuning（全層更新）する最小サンプル。
学習済みモデルは models/ft_full 以下に保存されます。
\"\"\"from training_utils import (
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
        output_dir="models/ft_full",
        num_train_epochs=1.0,
        learning_rate=5e-5,
    )

    ensure_dir(cfg.output_dir)

    tokenizer = load_base_tokenizer(cfg.model_name_or_path)
    model = load_base_model(cfg.model_name_or_path)

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
        run_name="tinyllama_full_ft",
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved full fine-tuned TinyLlama model to {cfg.output_dir}")


if __name__ == "__main__":
    main()
