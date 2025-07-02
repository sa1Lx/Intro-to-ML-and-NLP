from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import torch
import numpy as np
from transformers import pipeline

dataset = load_dataset("imdb")

small_train = dataset["train"].shuffle(seed=42).select(range(1000))  # 1k samples
small_test = dataset["test"].shuffle(seed=42).select(range(250))     # 250 samples

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",  # Pad shorter sequences
        truncation=True,       # Cut sequences >512 tokens
        max_length=512         # BERT's max input length
    )

tokenized_dataset_train = small_train.map(tokenize, batched=True)
tokenized_dataset_test = small_test.map(tokenize, batched=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        **metric.compute(predictions=predictions, references=labels),
        **f1_metric.compute(predictions=predictions, references=labels)
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    compute_metrics=compute_metrics,
)

trainer.train()

# Step 7: Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Step 8: Test inference
sample_text = "This movie was not fantastic!"
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("Sample prediction:", pipe(sample_text)) # Label_1: Positive, Label_0: Negative