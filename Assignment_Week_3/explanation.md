1. ***Step 1***: Load the IMDb dataset using the `datasets` library.

```python
dataset = load_dataset("imdb")
``` 
The IMDb dataset comes pre-labeled with 25K positive/negative reviews.

2. ***Step 2:***: Tokenize with BERT tokenizer

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
```
BERT needs text converted to numerical IDs. We use BertTokenizer to split text into subwords and pad/truncate sequences.
Returns input_ids (token IDs) and attention_mask (to ignore padding).

