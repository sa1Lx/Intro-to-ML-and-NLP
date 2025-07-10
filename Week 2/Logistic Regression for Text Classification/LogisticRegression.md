# Introduction to Text Classification Using Logistic Regression

Introduction to Logistic Regression: [StatQuest with Josh Starmer](https://www.youtube.com/watch?v=yIYKR4sgzI8)

Logistic Regression is a statistical method used for binary classification problems. It predicts the probability of a binary outcome based on one or more predictor variables. The output is transformed using the logistic function, which maps any real-valued number into the range [0, 1]. This makes it suitable for predicting probabilities.

Logistic regression does not use the least squares method or R-squared for model evaluation. Instead, it relies on maximum likelihood estimation to find the best-fitting curve. This involves calculating the likelihood of observing the actual data for a given curve, shifting the curve repeatedly, and selecting the one that maximizes this likelihood. Logistic regression can handle both continuous and discrete predictor variables, and it is also useful for assessing which variables contribute meaningfully to classification.

For the maths, refer [ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html).

# Text Classification with Logistic Regression

For implementation for text classification from scratch without use of any high level libraries, refer [Logistic Regression from Scratch](https://github.com/YukihoKirihara/Text-Classification-by-a-Logistic-Regression-Model-from-Scratch).