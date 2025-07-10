# Intro-to-ML-and-NLP

## About this Repository

This repository was developed by [sa1Lx](https://github.com/sa1Lx) as part of the "Introduction to Machine Learning and Natural Language Processing" course offered by IIT Bombay Learner Space. It contains well-documented implementations of core [NLP and ML concepts](#course-structure), including 2 major projects:
* [A Next Word Prediction model using GPT-2](Week_4_Final_Project/)
* [A Sentiment Analysis model using BERT](Assignment_Week%203/Sentiment%20Classifier_BERT/)

The repository is intended as both a learning archive and a reference for anyone looking to understand and build foundational NLP applications.

# Course Structure

## [Week 1](<Week 1/>)

Foundations of NLP & Data Processing

- [Python Libraries](<Week 1/Python%20Modules/>)

  * [NumPy](<Week 1/Python%20Modules/numpy.md>)
  * [Pandas](<Week 1/Python%20Modules/pandas.md>)
  * [Matplotlib](<Week 1/Python%20Modules/matplotlib.md>)
  * [Scikit-learn](<Week 1\Python Modules\scikit_learn.md>)
  * [NLTK](<Week 1/Python%20Modules/nltk.md>)

- [Regex (Regular Expressions)](<Week 1/Regex/regex.md>)

- [Introduction to NLP](<Week 1/NLP%20Pipeline/nlp_pipeline.md>)
    * Tokenization
    * Preprocessing 
    * Vectorization 
    * Modeling
    * Evaluation  

## [Week 2](<Week 2/>)

Classical NLP Modeling

  - [Text Vectorization](<Week 2/Text%20Vectorization/>)  
      * [Bag of Words (BoW)](<Week 2/Text%20Vectorization/CountVectorizer.ipynb>)
      * [TF-IDF](<Week 2/Text%20Vectorization/TfidfVectorizer.ipynb>)

  - [Logistic Regression for Text Classification](<Week 2/Logistic%20Regression%20for%20Text%20Classification/>)
    * Training and evaluating a classifier for sentiment analysis or spam detection.
    * Metrics: Accuracy, Precision, Recall, F1-score.  
    * Hands-on: train a TF-IDF + Logistic Regression model.  

  - [Word & Sentence Embeddings](<Week 2/Word%20&%20Sentence%20Embeddings/>)
    * Limitations of one-hot and TF-IDF.  
    * Word2Vec, GloVe, and FastText.  
    * Sentence embeddings (e.g., Sentence-BERT).  

## [Week 3](<Week 3/>)

Deep Learning & Modern NLP

  - [Transformers](<Week 3/Transformers/>)
    * [RNN](<Week 3/Transformers/RNN.md>)
    * [LSTM](<Week 3/Transformers/LSTM.md>)
    * [Encoder-Decoder Architecture](<Week 3/Transformers/Encoder_Decoder.md>)
    * [Attention Mechanism](<Week 3/Transformers/Attention_Mechanism.md>)
    * [Transformer Architecture](<Week 3/Transformers/Transformers.md>)

  - [Hugging Face Library](<Week 3/Hugging%20Face%20Library/>)
    * Using the transformers library.
    * Tokenization, loading pretrained models, pipelines (e.g., pipeline('sentiment-analysis')).
    * Fine-tuning on custom data using Trainer API.

  - [Overview of Generative Adversarial Networks](<Week 3/GAN/GAN.md>)
    * Understanding GANs and their applications in NLP.
    * Differences between GANs and traditional models.

  - Bonus: Diffusion in NLP [TBD]
    Overview of Diffusion Models.
    How these are adapted for text generation.

## [Week 4](Week_4_Final_Project/)

  - Final Project
      * Problem Statement: Implementation of a Next Word Prediction Model using GPT-2.
      * Frontend using Streamlit.


