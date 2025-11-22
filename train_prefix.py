# train_prefix.py
\"\"\"TinyLlama-1.1B-Chat-v1.0 に Prefix Tuning を適用するサンプル。
Prefix アダプタは models/ft_prefix 以下に保存されます。
\"\"\"from peft import PrefixTuningConfig, get_peft_model
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
        output_dir="models/ft_prefix",
        num_train_epochs=1.0,
        learning_rate=2e-4,
    )

    ensure_dir(cfg.output_dir)

    tokenizer = load_base_tokenizer(cfg.model_name_or_path)
    base_model = load_base_model(cfg.model_name_or_path)

    prefix_cfg = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
    )

    model = get_peft_model(base_model, prefix_cfg)
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
        run_name="tinyllama_prefix_ft",
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved TinyLlama Prefix adapter to {cfg.output_dir}")


if __name__ == "__main__":
    main()
