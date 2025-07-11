# Embeddings

Embeddings are vector representations of text (words, phrases, or sentences) that capture semantic meaning in a way that machines can understand. Instead of working with raw text, we transform it into dense vectors (usually floating-point numbers) that encode similarity and relationships.

A. [Word2Vec (Word Embedding)](#1-word2vec-word-embedding)
B. [Avg Word2Vec (Sentence Embedding)](#2-avg-word2vec-sentence-embedding)
C. [BERT (Sentence Embedding)](#3-bert-sentence-embedding)

## Word Embeddings

Word embeddings represent individual words as vectors. Words with similar meanings are mapped to nearby points in vector space.

| **Embedding Method**   | **Key Features**                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| **Word2Vec (Google)**  | Learns word vectors using context (CBOW and Skip-gram models).                  |
| **GloVe (Stanford)**   | Uses word co-occurrence counts across a large corpus. Captures global statistics.|
| **FastText (Facebook)**| Uses subword (character n-gram) information. Helps with rare/OOV words.         |

### Pros

* Captures semantic and syntactic relations.

* Works well for many NLP tasks like sentiment analysis, NER.

### Cons

* Fixed meaning for each word (e.g., "bank" of river vs. bank as financial institution).

* Doesn’t consider word order or context

## Sentence Embeddings

Sentence embeddings represent entire sentences (or even paragraphs) as single vectors. These embeddings take into account:

a. Word order
b. Context
c. Syntax & semantics

| **Model**                 | **Key Features**                                                                                      |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| **Universal Sentence Encoder (USE)** | Pretrained by Google. <br> Converts sentences to 512-dimensional vectors.                           |
| **Sentence-BERT (SBERT)**           | A modification of BERT designed for sentence-level embeddings. <br> Effective for similarity and semantic search tasks. |
| **InferSent**                        | Trained on natural language inference (NLI) data. <br> Encodes sentences with general-purpose meaning. |


### Pros:
* Encodes full sentence meaning and context.

* Useful for tasks like semantic similarity, document classification, retrieval, etc.

### Limitations:
* Computationally heavier than word embeddings.

* May require fine-tuning for domain-specific tasks


## 1. Word2Vec (Word Embedding)

Word2Vec is a popular **word embedding** technique developed by Google in 2013. It represents words as **dense vectors** in a continuous vector space where semantic similarity is captured using context in a large text corpus.

### CBOW & Skip-gram

Word2Vec uses a shallow neural network with two main variants:

- **CBOW (Continuous Bag of Words):** Predicts the current word from surrounding context.
- **Skip-gram:** Predicts surrounding context words given the current word.

### How it works:

- Similar words have similar vectors
- Words used in similar contexts are close in the vector space
- Vectors are typically 100–300 dimensions

## 2. Avg Word2Vec (Sentence Embedding)

Avg Word2Vec is a simple yet effective method to convert a sentence or document into a single fixed-size vector, using pre-trained word vectors.

### How it works:

* Split the sentence into words.
* Look up the Word2Vec vector for each word.
* Average all the vectors (i.e., element-wise mean).

### Pros:

* Super simple to implement
* Fast and efficient
* Doesn’t capture word order or contextual meaning
* Best for: baseline models, quick semantic similarity, etc.

## 3. BERT (Sentence Embedding)

BERT is a transformer-based model developed by Google AI in 2018. It’s trained to understand the context of a word in a sentence in both directions — left and right — making it bidirectional.

### Pretraining Tasks
* **Masked Language Modeling (MLM):**
Random words are masked → BERT learns to predict them

* **Next Sentence Prediction (NSP):**
Learns sentence relationships

### Limitations
* Large and slow
* Not ideal for sentence-level tasks → use SBERT instead
* Replaced by newer models like RoBERTa, DistilBERT, DeBERTa, GPT, etc