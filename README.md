# Intro-to-ML-and-NLP

## About this Repository

This repository was developed by [sa1Lx](https://github.com/sa1Lx) as part of the "Introduction to Machine Learning and Natural Language Processing" course offered by IIT Bombay Learner Space. It contains well-documented implementations of core [NLP and ML concepts](#course-structure), including 2 major projects:
* [A Next Word Prediction model using GPT-2](Week_4_Final_Project/)
* [A Sentiment Analysis model using BERT](Assignment_Week_3/Sentiment%20Classifier_BERT/)

The repository is intended as both a learning archive and a reference for anyone looking to understand and build foundational NLP applications.

# Course Structure

## [Week 1](Week_1_Content/)

Foundations of NLP & Data Processing

- [Python Libraries](Week_1_Content/Python%20Modules/)

  * [NumPy](Week_1_Content/Python%20Modules/numpy.md)
  * [Pandas](Week_1_Content/Python%20Modules/pandas.md)
  * [Matplotlib](Week_1_Content/Python%20Modules/matplotlib.md)
  * [Scikit-learn](Week_1_Content/Python%20Modules/scikit_learn.md)
  * [NLTK](Week_1_Content/Python%20Modules/nltk.md)

- [Regex (Regular Expressions)](Week_1_Content/Regex/regex.md)  

- [Introduction to NLP](Week_1_Content/NLP%20Pipeline/nlp_pipeline.md)
    * Tokenization 
    * Preprocessing 
    * Vectorization 
    * Modeling
    * Evaluation  

## Week 2

Classical NLP Modeling

  - Text Vectorization  
    TF-IDF (Term Frequency-Inverse Document Frequency):  
    Concept of bag-of-words vs TF-IDF.  

  - Logistic Regression for Text Classification  
    Training and evaluating a classifier for sentiment analysis or spam detection.  
    Metrics: Accuracy, Precision, Recall, F1-score.  
    Hands-on: train a TF-IDF + Logistic Regression model.  

  - Word & Sentence Embeddings  
    Limitations of one-hot and TF-IDF.  
    Word2Vec, GloVe, and FastText.  
    Sentence embeddings (e.g., Sentence-BERT).  

## [Week 3](Week_3_Content/)

Deep Learning & Modern NLP

  - [Transformers](Week_3_Content/Transformers)
    * [RNN](Week_3_Content/Transformers/RNN.md)
    * [LSTM](Week_3_Content/Transformers/LSTM.md)
    * [Encoder-Decoder Architecture](Week_3_Content/Transformers/Encoder_Decoder.md)
    * [Attention Mechanism](Week_3_Content/Transformers/Attention_Mechanism.md)
    * [Transformer Architecture](Week_3_Content/Transformers/Transformers.md)

  - [Hugging Face Library](Week_3_Content/Hugging%20Face%20Library/)
    * Using the transformers library.
    * Tokenization, loading pretrained models, pipelines (e.g., pipeline('sentiment-analysis')).
    * Fine-tuning on custom data using Trainer API.

  - [Overview of Generative Adversarial Networks](Week_3_Content/GAN/GAN.md)
    * Understanding GANs and their applications in NLP.
    * Differences between GANs and traditional models.

  - Bonus: Diffusion in NLP [TBD]
    Overview of Diffusion Models.
    How these are adapted for text generation.

## [Week 4](Week_4_Final_Project/)

  - Final Project
      * Problem Statement: Implementation of a Next Word Prediction Model using GPT-2.
      * Frontend using Streamlit.


