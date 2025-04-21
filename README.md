# ECE-GY-7123-DL — AGNews Text Classification with LoRA

---

## **Overview**  
This project implements a **LoRA-adapted RoBERTa model** for **AGNews topic classification**, targeting high accuracy under a strict **1M trainable parameter constraint**. Leveraging **Low-Rank Adaptation (LoRA)**, we fine-tune only a small subset of weights while freezing the base transformer model.

The training pipeline is designed with **efficient evaluation, adaptive learning rate scheduling**, and thorough monitoring of accuracy, loss, and model complexity. Our final model achieves a **validation accuracy of 88.42% using only 667,396 trainable parameters**, making it suitable for deployment in low-resource environments.

---

## **Dataset**  
We use the **AGNews dataset**, which contains approximately **120,000 news headlines and descriptions**, categorized into four classes:
- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Each sample includes a title and short description, concatenated and tokenized using RoBERTa’s Byte-Pair Encoding (BPE). The dataset is split into 90% training and 10% validation with stratification.

---

## **Model Architecture**

We build upon the pretrained `roberta-base` transformer architecture and adapt it using **Low-Rank Adaptation (LoRA)**, which injects trainable low-rank matrices into specific layers while freezing the base model weights. This enables parameter-efficient fine-tuning.

### LoRA Configuration:
- **Target Modules**: `query`, `key`
- **Rank (`r`)**: 4  
- **Scaling Factor (`alpha`)**: 8  
- **Dropout**: 0.1  
- **Trainable Parameters**: 667,396

The classifier head projects the `[CLS]` token to 4 output classes. Only the LoRA adapters and classification head are trained.

---

## **Installation**

Install all required Python packages:

```bash
pip install torch transformers datasets peft evaluate matplotlib scikit-learn
```


Sure! Here's the **Usage**, **Configuration**, and **Results** sections in clean Markdown format, ready to plug into your `README.md`:

---

### 🧪 **Usage**

#### 1. **Run Training**

```bash
python run_lora_training.py
```

This script:
- Downloads and tokenizes the AGNews dataset
- Applies LoRA to a `roberta-base` model
- Trains using HuggingFace's Trainer API
- Logs and saves performance metrics and plots

#### 2. **Generated Outputs**

| File/Folder | Description |
|-------------|-------------|
| `trained_models/final_model/` | Final model checkpoint and tokenizer |
| `accuracy_curve.png`         | Evaluation accuracy vs steps plot |
| `loss_curve.png`             | Training and validation loss vs steps plot |
| Console Output               | Prints final validation accuracy and parameter count |

---

### ⚙️ **Configuration**

Core configuration (defined in-code but optionally externalizable):

```yaml
model:
  base_model: roberta-base
  target_modules: [query, key]
  lora_rank: 4
  lora_alpha: 8
  lora_dropout: 0.1

training:
  max_seq_len: 80
  batch_size_train: 16
  batch_size_eval: 32
  learning_rate: 2e-5
  epochs: 3
  eval_steps: 250
  scheduler: linear
  warmup_ratio: 0.1
  freeze_base_model: true
```

You can modify these inside the script or move them to a `config.yaml` file for modular control.

---

### **Results**

| **Metric**                | **Value**      |
|---------------------------|----------------|
| Validation Accuracy       | **88.42%**     |
| Trainable Parameters      | 667,396        |
| LoRA Rank (`r`)           | 4              |
| LoRA Alpha (`α`)          | 8              |
| LoRA Dropout              | 0.1            |
| Target Modules            | `query`, `key` |
| Max Sequence Length       | 80 tokens      |
| Batch Size (Train / Eval) | 16 / 32        |
| Training Epochs           | 3              |
| Evaluation Interval       | Every 250 steps|

Validation accuracy peaked at 88.42% with strong convergence, low overfitting, and efficient LoRA adaptation—all while staying under the 1M trainable parameter cap.

---

Absolutely! Here's the **Conclusion**, **Project Report**, and **Acknowledgments** sections written in Markdown, formatted to match the rest of your `README.md`:

---

### ✅ **Conclusion**

This project demonstrates that **Low-Rank Adaptation (LoRA)** enables effective and scalable fine-tuning of large transformer models such as RoBERTa in resource-constrained environments. By introducing trainable adapters only in selected self-attention layers (`query`, `key`), and freezing the rest of the base model, we achieved a validation accuracy of **88.42%** using just **667,396 trainable parameters**—well within the 1 million parameter budget.

Careful design choices in adapter configuration (rank, alpha, dropout), tokenization, and evaluation strategy contributed to fast convergence and strong generalization. The result is a compact yet accurate model suitable for real-world deployment on low-power or memory-limited devices.

---

### 📄 **Project Report**

A full report outlining our motivation, architecture, training pipeline, and results—including diagrams and plots—is available in the project report given in the repository.

This PDF includes:
- Abstract & Introduction  
- Dataset processing strategy  
- Detailed LoRA configuration  
- Training schedule and optimizer choices  
- Evaluation metrics and result plots  
- Final analysis, discussion, and conclusion  

---

### 🙏 **Acknowledgments**

This project uses open-source tools and datasets. We thank the authors and maintainers of:

- 🤗 [HuggingFace Transformers](https://github.com/huggingface/transformers) — for model definitions and training utilities  
- 🧠 [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) — for LoRA integration  
- 📚 [AGNews Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) — for providing the classification benchmark  
- 🎓 NYU Tandon’s ECE-GY 7123 Deep Learning course — for designing a research-focused, application-driven project framework
- Chat GPT has been instrumental in reformatting my codes into cleaner versions and debugging, along with help in making the language and structure of the project report better.

---

