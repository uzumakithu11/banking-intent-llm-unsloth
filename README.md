# Intent Classification with Llama (Unsloth)

## Overview

This project fine-tunes a LLaMA model using Unsloth for intent classification on banking queries.

---

## Environment Setup

### 1. Clone repository

```bash
git clone http://github.com/uzumakithu11/banking-intent-llm-unsloth
cd banking-intent-llm-unsloth
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

Place your dataset at:

```
sample_data
```

Run the command to download dataset and preprocess it

```bash
python scripts/preprocess_data.py
```
After finishing preprocessing dataset, train.csv and test.csv will appear in ../sample_data


---

## Training

### Run training

```bash
train.sh
```

Or directly:

```bash
python train.py
```

### Config file

Training uses:

```
configs/train.yaml
```

You can modify:

* model name
* batch size
* learning rate
* LoRA parameters

---

## Output

After training, model is saved at:

```
outputs/checkpoint-final
```

Includes:

* fine-tuned model
* tokenizer

---

## Inference

```bash
inference.sh
```

Or Python:

```python
python inference.py
```

## Dependencies

Main libraries:

* unsloth
* transformers
* trl
* datasets
* bitsandbytes
