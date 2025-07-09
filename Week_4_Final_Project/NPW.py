# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
model.to("cuda") # Model loaded below first
print("Model moved to CUDA successfully.")
print(torch.cuda.memory_allocated(0) / (1024 ** 2))
print(torch.cuda.memory_reserved(0) / (1024 ** 2))

# %%
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("wikitext", "wikitext-2-v1") # was too large to train so just used for testing

# %%
total_size = sum(split.num_bytes for split in ds_builder.info.splits.values()) / (1024 ** 2)
print(f"Total dataset size: ~{total_size:.2f} MB")
print("Dataset description:", ds_builder.info.description)
print("\nFeatures:\n", ds_builder.info.features)
print("\nSplits:\n", list(ds_builder.info.splits.keys()))


# %%
from datasets import load_dataset_builder
ds_builder_alt = load_dataset_builder("stas/openwebtext-10k")

# %%
total_size = sum(split.num_bytes for split in ds_builder_alt.info.splits.values()) / (1024 ** 2)
print(f"Total dataset size: ~{total_size:.2f} MB")
print("Dataset description:", ds_builder_alt.info.description)
print("\nFeatures:\n", ds_builder_alt.info.features)
print("\nSplits:\n", list(ds_builder_alt.info.splits.keys()))

# %%
ds_builder.download_and_prepare()
ds = ds_builder.as_dataset()

# %%
ds_builder_alt.download_and_prepare()
ds_alt = ds_builder_alt.as_dataset()

# %%
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%
tokenizer.pad_token = tokenizer.eos_token
def tokenize(examples):
    return tokenizer(examples["text"], max_length=256, padding="max_length", truncation=True)

# %%
def tokenize_test(example): # for measuring performance
    tokens = tokenizer(
        example["text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()

# %%
tokenized_dataset = ds.map(tokenize_test, batched=True)

# %%
tokenized_dataset_alt = ds_alt.map(tokenize, batched=True)

# %%
tokenized_dataset_alt = tokenized_dataset_alt["train"].train_test_split(test_size=0.2, seed=42, shuffle=True)

# %%
from sklearn.metrics import top_k_accuracy_score
import torch
import numpy as np
import math

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

    return {
        "perplexity": perplexity,
        "top5_accuracy": topk_acc
    }

# %%
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

# %%
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./NWP_results",
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
    train_dataset=tokenized_dataset_alt["train"],
    eval_dataset=tokenized_dataset_alt["test"],
    data_collator=data_collator,
)

trainer.train()

# %%
model.save_pretrained("./gpt2-finetuned-nwp")
tokenizer.save_pretrained("./gpt2-finetuned-nwp")

# %%
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-nwp")
tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned-nwp")
model.eval() # Set model to evaluation mode
model.to("cuda")

# %%
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

test_ds = tokenized_dataset["test"].shuffle(seed=42).select(range(100))

training_args = TrainingArguments(output_dir="./dummy", per_device_eval_batch_size=1, fp16=True, eval_accumulation_steps=8)

trainer = Trainer(
    model=model,
    args=training_args,
)

# Predict on test set
outputs = trainer.predict(test_ds)

# %%
assert outputs.predictions is not None, "Predictions are None"
assert outputs.label_ids is not None, "Label IDs are None"
metrics = compute_metrics((outputs.predictions, outputs.label_ids))
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# %%
prompt = "I like watching"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs["input_ids"], max_new_tokens=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


