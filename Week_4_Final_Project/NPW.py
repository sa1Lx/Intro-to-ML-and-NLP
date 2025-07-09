from datasets import load_dataset_builder
ds_builder = load_dataset_builder("wikitext", "wikitext-2-v1")

total_size = sum(split.num_bytes for split in ds_builder.info.splits.values()) / (1024 ** 2)
print(f"Total dataset size: ~{total_size:.2f} MB")
print("Dataset description:", ds_builder.info.description)
print("\nFeatures:\n", ds_builder.info.features)
print("\nSplits:\n", list(ds_builder.info.splits.keys()))

ds_builder.download_and_prepare()
ds = ds_builder.as_dataset()

from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
model.to("cuda") # Model loaded below first
print("Model moved to CUDA successfully.")
print(torch.cuda.memory_allocated(0) / (1024 ** 2))
print(torch.cuda.memory_reserved(0) / (1024 ** 2))

tokenizer.pad_token = tokenizer.eos_token
def tokenize(examples):
    return tokenizer(examples["text"], max_length=256, padding="max_length", truncation=True)

tokenized_dataset = ds.map(tokenize, batched=True)

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./NWP_final_results",
    eval_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./gpt2-finetuned-nwp-final")
tokenizer.save_pretrained("./gpt2-finetuned-nwp-final")

from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-nwp-final")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned-nwp-final")
model.eval() # Set model to evaluation mode

test_ds = tokenized_dataset["test"].shuffle(seed=42).range(100)  # Use a smaller subset for testing

from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

training_args = TrainingArguments(output_dir="./dummy", per_device_eval_batch_size=1, fp16=True, eval_accumulation_steps=8, remove_unused_columns=False)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)

# Predict on test set
outputs = trainer.predict(test_ds)

from sklearn.metrics import top_k_accuracy_score
import torch
import numpy as np
import math
from evaluate import load

perplexity_metric = load("perplexity")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Shift so model predicts token t+1
    shift_logits = torch.tensor(logits)[..., :-1, :].contiguous()
    shift_labels = torch.tensor(labels)[..., 1:].contiguous()

    # Flatten the tensors
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Mask out padding
    valid = shift_labels != -100
    y_true = shift_labels[valid].numpy()
    y_pred = shift_logits[valid].numpy()

    # Perplexity (cross-entropy)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(y_pred), torch.tensor(y_true))
    perplexity = math.exp(loss.item())

    # Top-k accuracy
    topk_acc = top_k_accuracy_score(y_true, y_pred, k=5, labels=list(range(50257))) # GPT-2 vocab size is 50257

    # perplexity

    results = perplexity_metric.compute(predictions=y_pred, references=y_true)

    return {
        "perplexity": perplexity,
        "top5_accuracy": topk_acc
    }

assert outputs.predictions is not None, "Predictions are None"
assert outputs.label_ids is not None, "Label IDs are None"
metrics = compute_metrics((outputs.predictions, outputs.label_ids))
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

from evaluate import load

perplexity = load("perplexity")

raw_test_texts = test_ds["text"]

raw_test_texts = [t for t in test_ds["text"] if t.strip() != ""]

results = perplexity.compute(
    predictions=raw_test_texts,
    model_id="./gpt2-finetuned-nwp-final",
    device="cpu"
)

print("Perplexity:", results["perplexity"])

