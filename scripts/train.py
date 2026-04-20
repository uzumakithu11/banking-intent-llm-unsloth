import yaml


def train(config_path: str = "configs/train.yaml"):
    import pandas as pd
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # ===== Load config =====
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ===== Model =====
    model_cfg = config["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    # ===== LoRA =====
    lora_cfg = config["lora"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    # ===== Data =====
    data_cfg = config["data"]
    prompt_template = config["prompt"]["template"]

    df = pd.read_csv(data_cfg["train_path"])

    EOS_TOKEN = tokenizer.eos_token

    def format_data(df):
        texts = []
        for _, row in df.iterrows():
            text = prompt_template.format(
                row[data_cfg["text_column"]],
                row[data_cfg["label_column"]],
            ) + EOS_TOKEN
            texts.append(text)
        return Dataset.from_dict({"text": texts})

    dataset = format_data(df)

    # ===== Trainer =====
    train_cfg = config["training"]
    opt_cfg = config["optimizer"]
    prec_cfg = config["precision"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        packing=config["packing"],

        args=SFTConfig(
            per_device_train_batch_size=train_cfg["batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            warmup_steps=train_cfg["warmup_steps"],
            num_train_epochs=train_cfg["num_epochs"],
            learning_rate=train_cfg["learning_rate"],

            logging_steps=train_cfg["logging_steps"],
            optim=opt_cfg["type"],
            weight_decay=opt_cfg["weight_decay"],
            lr_scheduler_type=opt_cfg["lr_scheduler"],

            fp16=prec_cfg["fp16"],
            bf16=prec_cfg["bf16"],
            max_grad_norm=config["max_grad_norm"],

            seed=config["seed"],
            output_dir=train_cfg["output_dir"],
            report_to="none",
        ),
    )

    # ===== Train =====
    trainer.train()

    # ===== Save =====
    output_dir = train_cfg["output_dir"] + "/checkpoint-final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
