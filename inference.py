import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

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


# ===== TEST LOCAL =====
if __name__ == "__main__":
    model = IntentClassification("configs/inference.yaml")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")

    print("Prediction:", model(query))
