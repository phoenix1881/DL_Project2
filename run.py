import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig
import evaluate
import torch
from datasets import Dataset

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -----------------------------
# Configuration Constants
# -----------------------------
MODEL_ID = "roberta-base"
SAVE_DIR = "trained_models/lora_roberta_random"
MAX_LEN = 80 
BATCH_TRAIN = 16  
BATCH_EVAL = 32  
N_EPOCHS = 3     
LR = 2e-5         
EVAL_EVERY = 250  
SAVE_EVERY = 1000 
SHOULD_FREEZE = True
 
LOG_DIR = "logs_lora_random"



# -----------------------------
# Logger Setup (Console + File)
# -----------------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_DIR, f"run_{timestamp}.log")

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
log.addHandler(file_handler)
log.addHandler(console_handler)

# -----------------------------
# Dataset Preparation (manual)
# -----------------------------
def load_agnews_csv():
    agnews_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    local_path = "ag_news_train.csv"

    if not os.path.exists(local_path):
        log.info("Downloading AG News training data...")
        df = pd.read_csv(agnews_url, header=None, names=["label", "title", "description"])
        df.to_csv(local_path, index=False)
    else:
        df = pd.read_csv(local_path)

    df["text"] = df["title"] + " " + df["description"]
    df["label"] = df["label"] - 1 
    return df[["text", "label"]]

def get_datasets():
    df = load_agnews_csv()
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    return tokenizer, train_dataset, val_dataset

# -----------------------------
# LoRA & Trainer Setup
# -----------------------------
def enhance_with_lora(model):
    config = LoraConfig(
        r=4,                    
        lora_alpha=8,          
        lora_dropout=0.1,        
        bias="none",              
        target_modules=["query", "key"],  
        task_type="SEQ_CLS"
    )
    return get_peft_model(model, config)


def selectively_freeze(model):
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

def accuracy_metric():
    scorer = evaluate.load("accuracy")
    def compute(p):
        logits, labels = p
        preds = np.argmax(logits, axis=1)
        return scorer.compute(predictions=preds, references=labels)
    return compute

def setup_training(model, tokenizer, train_data, val_data):
    args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=N_EPOCHS,
        learning_rate=LR,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_EVERY,
        save_steps=SAVE_EVERY,
        logging_dir=LOG_DIR,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear"
    )

    callbacks = []

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=accuracy_metric(),
        callbacks=callbacks
    )
    return trainer

# -----------------------------
# Main Pipeline
# -----------------------------
def execute_pipeline():
    tokenizer, train_data, val_data = get_datasets()

    log.info("Loading model and applying LoRA...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=4,
        id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
    )
    model = enhance_with_lora(model)

    if SHOULD_FREEZE:
        log.info("Freezing base model parameters except LoRA + classifier...")
        selectively_freeze(model)

    trainer = setup_training(model, tokenizer, train_data, val_data)

    log.info("Starting training...")
    trainer.train()

    log.info("Evaluating on validation set...")
    metrics = trainer.evaluate()
    log.info(f"Final Evaluation Metrics: {metrics}")

    log.info("Saving model and tokenizer...")
    model.save_pretrained(os.path.join(SAVE_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "final_model"))

    log.info("Training pipeline completed successfully!")


    log.info("Evaluating on validation set...")
    metrics = trainer.evaluate()
    log.info(f"Final Evaluation Metrics: {metrics}")
    print(f"\n Final Validation Accuracy: {metrics['eval_accuracy']:.4f}")

    # Count and log trainable parameters
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = count_trainable_params(model)
    log.info(f" Total Trainable Parameters: {trainable_params}")
    print(f"\n Total Trainable Parameters: {trainable_params}")

    # Generate plots
    import matplotlib.pyplot as plt

    log.info("Generating plots...")
    logs = trainer.state.log_history
    steps, train_loss, eval_loss, eval_acc = [], [], [], []

    for entry in logs:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
            steps.append(entry["step"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "eval_accuracy" in entry:
            eval_acc.append(entry["eval_accuracy"])

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
    plt.plot(steps[:len(eval_loss)], eval_loss, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(eval_acc)], eval_acc, label="Eval Accuracy", color='green')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy vs Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png")
    plt.show()


if __name__ == "__main__":
    execute_pipeline()
