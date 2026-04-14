import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ======================
# 1. Clean text
# ======================
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # remove URL
    text = re.sub(r"[^a-zA-Z0-9\s'\-]", " ", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


# ======================
# 2. Load dataset
# ======================
def load_banking77():
    dataset = load_dataset("PolyAI/banking77")
    train_data = dataset["train"]
    test_data = dataset["test"]
    label_names = dataset["train"].features["label"].names
    return train_data, test_data, label_names


# ======================
# 3. Convert to DataFrame
# ======================
def to_dataframe(data, label_names):
    texts = [clean_text(x["text"]) for x in data]
    labels = [label_names[x["label"]] for x in data]

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })
    return df


# ======================
# 4. Sample subset
# ======================

def sample_subset(df, n_per_label=40):
    return df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_per_label), random_state=42)
    )


# ======================
# 5. Main pipeline
# ======================
def preprocess(output_dir = "../sample_data", n_samples=3000):
    train_data, test_data, label_names = load_banking77()

    df_train = to_dataframe(train_data, label_names)
    df_test = to_dataframe(test_data, label_names)

    # Gộp lại rồi sample
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    df_subset = sample_subset(df_all, n_per_label=40)

    # Split lại train/test
    train_df, test_df = train_test_split(
        df_subset,
        test_size=0.2,
        random_state=42,
        stratify=df_subset["label"]
    )

    # Save
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("Done!")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")


# ======================
# 6. Run
# ======================
if __name__ == "__main__":
    preprocess(n_samples=3000)