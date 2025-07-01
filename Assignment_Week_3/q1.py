from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch
import numpy as np

# Step 1: Load IMDb dataset
dataset = load_dataset("imdb")

# Step 2: Tokenize with BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",  # Pad shorter sequences
        truncation=True,       # Cut sequences >512 tokens
        max_length=512         # BERT's max input length
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
print("Tokenized dataset:", tokenized_dataset)

# Step 3: Prepare model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 4: Define metrics (accuracy & F1)
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Step 5: Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Enable GPU acceleration
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Step 6: Train!
trainer.train()

# Step 7: Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Step 8: Test inference
sample_text = "This movie was fantastic!"
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("Sample prediction:", pipe(sample_text))