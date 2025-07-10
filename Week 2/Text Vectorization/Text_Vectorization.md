# Text Vectorization

Text vectorization is a set of fundamental techniques for converting text into a numerical format that machine learning models can understand. <br>
Corpus: Collection of documents or text data. <br>
We will be comparing two techniques in this file:

* [Bag of Words (BoW)](#1-bag-of-words-bow) implemented as [`CountVectorizer`](CountVectorizer.ipynb) in `scikit-learn`
* [Term Frequency-Inverse Document Frequency (TF-IDF)](#2-term-frequency-inverse-document-frequency-tf-idf) implemented as [`TfidfVectorizer`](TfidfVectorizer.ipynb) in `scikit-learn`

## 1. Bag of Words (BoW)

The Bag of Words model represents text data as a collection of words, disregarding grammar and word order but keeping multiplicity. Each unique word in the corpus is treated as a feature, and the frequency of each word in a document is counted. It is a feature extraction technique that transforms text into a vector of word counts.

Can be extended as to visual elements too: BoVW (Bag of Visual Words)

### Usecases:

* Identifying spam emails
* Document Similarity
* Query Classification/Retrieval

### Pros

* Simple and easy to implement
* Explainable and interpretable

### Cons

* Compound Words (New York, Artificial Intelligence)
* Doesnt stress word correlation at all
* Polysemous words (bank, bank)
* Doesnt capture word order
* Sparsity (high dimensionality)

### Modifications

* **N-grams**: Instead of single words, use sequences of n words (bigrams, trigrams, etc.) to capture some context.
* **Text Normalization**: Apply techniques like stemming or lemmatization to reduce words to their base forms.

## 2. Term Frequency-Inverse Document Frequency (TF-IDF)

Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus). It combines two components:
* **Term Frequency (TF)**: Measures how frequently a term appears in a document. It is calculated as the number of times a term appears in a document divided by the total number of terms in that document.
* **Inverse Document Frequency (IDF)**: Measures how important a term is across the entire corpus. It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term. The logarithm is used to dampen the effect of very large or very small values, ensuring the IDF score scales appropriately.

The TF-IDF score is the product of these two components, giving higher weight to terms that are frequent in a document but rare across the corpus.

### Usecases:

* Document Classification
* Information Retrieval
* Filtering Stop Words
* Highlighting Unique Terms

### Pros

* Considers word importance across the corpus
* Reduces the impact of common words (stop words)

### Cons

* Still ignores word order
* Can be computationally expensive for large corpora
* Sparse representation
* Polysemous words (bank, bank)

### Modifications

Similar to BoW, TF-IDF can also be extended with n-grams and text normalization techniques to improve context capture and reduce sparsity.

# Comparison: Bag-of-Words vs. TF-IDF

| Feature                | Bag-of-Words (BoW)                                                  | Term Frequency-Inverse Document Frequency (TF-IDF)                                 |
| :--------------------- | :------------------------------------------------------------------ | :--------------------------------------------------------------------------------- |
| **Core Idea** | Counts the frequency of each word in a document.                    | Weighs words based on their frequency in a document and their rarity across all documents. |
| **Word Importance** | All words are treated equally. Common words can dominate.           | Gives higher weight to words that are frequent in a document but rare in the corpus. |
| **Context** | Does not capture the context or semantic meaning of words.          | Also does not capture context, but it can better identify important, topic-specific words. |
| **Vector Representation** | Vectors contain raw word counts or frequencies.                     | Vectors contain weighted scores ($TF \times IDF$ values) for each word.                      |
| **Use Case** | Good for simple text classification tasks where word counts are sufficient. | Better for more complex tasks like search engines and document clustering.        |


