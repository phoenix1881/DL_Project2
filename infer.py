import logging
import torch
import pickle
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_dir = "trained_models/lora_roberta_v2"
batch_size = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load the tokenizer and model
    logger.info("Loading tokenizer and model from fine-tuned checkpoint")
    tokenizer = RobertaTokenizer.from_pretrained(f'./{model_dir}/final_model')
    model = RobertaForSequenceClassification.from_pretrained(f'./{model_dir}/final_model', num_labels=4)
    model.to(device)
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Load custom test set from pickle file
    logger.info("Loading custom test dataset")
    with open("test_unlabelled.pkl", "rb") as f:
        custom_test_dataset = pickle.load(f)  # should be a HuggingFace Dataset object

    # Tokenization function
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    # Tokenize the dataset
    logger.info("Tokenizing custom test dataset")
    tokenized_dataset = custom_test_dataset.map(preprocess, batched=True, remove_columns=["text"])

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    all_predictions = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_predictions.append(preds.cpu())

    predicted_labels = torch.cat(all_predictions, dim=0)
    predicted_labels = predicted_labels.numpy()

    # # Set format for PyTorch
    # tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # # Use Trainer for prediction
    # trainer = Trainer(model=model)

    # logger.info("Running predictions on custom test set")
    # predictions = trainer.predict(tokenized_dataset)

    # predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Combine with original IDs
    ids = list(range(len(custom_test_dataset)))
    output = [{"ID": id_, "Label": label} for id_, label in zip(ids, predicted_labels)]

    # Save predictions to a CSV file
    output_df = pd.DataFrame(output)
    output_df.to_csv("predictions.csv", index=False)

    logger.info("Predictions saved to custom_test_predictions.csv")

if __name__ == "__main__":
    main()
