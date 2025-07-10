from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import torch
import numpy as np
from transformers import pipeline

# Load the IMDb dataset using the `datasets` library
dataset = load_dataset("imdb")

# For demonstration purposes, using a small subset of the dataset
small_train = dataset["train"].shuffle(seed=42).select(range(1000))  # 1k samples
small_test = dataset["test"].shuffle(seed=42).select(range(250))     # 250 samples

# Load the BERT tokenizer sequence classification
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

# Prepare model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define metrics (accuracy & F1)
metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        **metric.compute(predictions=predictions, references=labels),
        **f1_metric.compute(predictions=predictions, references=labels)
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model = BertForSequenceClassification.from_pretrained("E:/IITB/Learner Space 2025/Intro-to-ML-and-NLP/results/checkpoint-126")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained("E:/IITB/Learner Space 2025/Intro-to-ML-and-NLP/fine_tuned_bert")
tokenizer.save_pretrained("E:/IITB/Learner Space 2025/Intro-to-ML-and-NLP/fine_tuned_bert")

# Using sample text for prediction
sample_text = "This movie was not fantastic!"
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("Sample prediction:", pipe(sample_text)) # Label_1: Positive, Label_0: Negative