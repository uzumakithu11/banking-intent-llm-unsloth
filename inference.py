import yaml
import torch
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score

class IntentClassification:
    def __init__(self, model_path):
        # load config yaml
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.model_dir = config["model_path"]
        self.max_seq_length = config.get("max_seq_length", 512)

        # load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.model.eval()

    def __call__(self, message):
        prompt = f"""Classify the intent of the following banking query.

### Query:
{message}

### Intent:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # extract label
        try:
            predicted_label = decoded.split("Intent:")[-1].strip().split("\n")[0]
        except:
            predicted_label = decoded.strip()

        return predicted_label

    def evaluate(self, test_path):
        df = pd.read_csv(test_path)

        y_true = []
        y_pred = []

        for _, row in df.iterrows():
            text = row["text"]
            label = row["label"]

            pred = self(text)

            y_true.append(label)
            y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)

        print("\n===== EVALUATION RESULT =====")
        print(f"Accuracy: {acc:.4f}")

        return acc


# ===== TEST LOCAL =====
if __name__ == "__main__":
    model = IntentClassification("configs/inference.yaml")

    # CLI inference
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eval":
            test_path = sys.argv[2]
            model.evaluate(test_path)

        else:
            query = " ".join(sys.argv[1:])
            print("Prediction:", model(query))

    else:
        query = input("Enter query: ")
        print("Prediction:", model(query))
