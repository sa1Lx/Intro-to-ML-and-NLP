# SciKit Learn

More on [SciKit Learn](https://scikit-learn.org/stable/)

Some Features of SciKit Learn:
* ***[Preprocessing](#preprocessing)***: Transforming raw data into a format suitable for machine learning models
    - ***[Feature Scaling](#a-feature-scaling)***: Normalizing or standardizing the range of independent variables (features)
    - ***[Quantile Transformer](#b-quantile-transformer)***: Transforming features to follow a uniform or normal distribution
    - ***[One Hot Encoder](#c-one-hot-encoder)***: Converting categorical variables into a format that can be provided to ML algorithms
    - ***[Polynomial Features](#d-polynomial-features)***: Generating polynomial and interaction features
* ***[Pipelines](#pipeline-in-scikit-learn)***: Streamlining the workflow of preprocessing and model training
* ***[Grid SearchCV](#gridsearchcv)***: Hyperparameter tuning for model optimization

to install `pip install scikit-learn`<br>
to import:
        import sklearn  # General import (rarely used alone)

        # Import a classifier
        from sklearn.linear_model import LogisticRegression

        # For data splitting
        from sklearn.model_selection import train_test_split

        # For preprocessing
        from sklearn.preprocessing import StandardScaler

        # For metrics
        from sklearn.metrics import accuracy_score


## Basic SciKit Learn Example
                # Step 1: Import necessary modules
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error

                # Step 2: Sample data (feature and target)
                X = [[1], [2], [3], [4], [5]]      # Features (independent variable)
                y = [2, 4, 6, 8, 10]               # Target (dependent variable)

                # Step 3: Split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Step 4: Create and train (fit) the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Step 5: Make predictions
                predictions = model.predict(X_test)

                # Step 6: Evaluate the model
                mse = mean_squared_error(y_test, predictions)
                print("Predictions:", predictions)
                print("Mean Squared Error:", mse)

## Explanation

1. **Import necessary modules**: Import the required classes and functions from `sklearn`.
* `LinearRegression`: The machine learning model we’ll use to predict a continuous value.

* `train_test_split`: Function to divide your data into training and testing sets.

* `mean_squared_error`: A metric to measure how good the model’s predictions are.

2. **Sample Data**: 
* `X` contains your input features. Each inner list is one sample (e.g., [[1]], [[2]], etc.).
* `y` is the output or label you want the model to learn (in this case, it’s simply 2 times the input).

3. **Splitting The Dataset**:
* Divides the data: 80% for training and 20% for testing.
* `random_state=42` ensures that the split is reproducible. This means that every time you run the code with the same `random_state`, you will get the same train-test split.
* `X_train`, `y_train`: Used to train the model.
* `X_test`, `y_test`: Used to test the model after training.

4. **Creating and Training the Model**:
* `model = LinearRegression()` creates a Linear Regression model object.
* `model.fit(X_train, y_train)` trains the model using the training data so it learns the relationship between `X` and `y`.

5. **Making Predictions**:
* `model.predict(X_test)` uses the trained model to predict values for the unseen test data.

6. **Evaluating the Model**:
* `mean_squared_error()` calculates the average squared difference between predicted and actual values.
* A lower MSE means the model is performing well.

# Preprocessing
Preprocessing in Scikit-learn involves transforming raw data into a format suitable for machine learning models. It helps improve model accuracy and performance.

## A. Feature Scaling
Feature Scaling is the process of normalizing or standardizing the range of independent variables (features). It ensures that no feature dominates others due to its scale.

```bash
preprocessor = 'required model'
X_trans = preprocessor.fit_transform(X)
```

### 1. Standardization
Standardization rescales features to have a mean of 0 and a standard deviation of 1. It is useful when features have different units or scales.

$$
z = \frac{x - \mu}{\sigma}
$$

​
 
`Result`: Mean = 0, Std = 1

### Tool:
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Min-Max Scaling
Formula:

$$
x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$



 
`Result`: Scales features to a [0, 1] range.

### Tool:

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## B. Quantile Transformer
Quantile Transformer is used to transform features so that they follow a uniform or normal (Gaussian) distribution. It works by ranking the data and mapping it to a specified distribution.

- Ranks the data.

- Maps the ranks to a desired distribution:

  - Uniform (default)

   - Normal (useful for linear models)

```
from sklearn.preprocessing import QuantileTransformer
import numpy as np

# Example data
X = np.array([[1.0], [2.0], [2.5], [3.0], [5.0]])

# Create the transformer
qt = QuantileTransformer(output_distribution='normal')  # or 'uniform'

# Fit and transform
X_trans = qt.fit_transform(X)

print(X_trans)
```

Parameters
- `output_distribution`=`'uniform'` (default) or `'normal'`

- `n_quantiles`: Number of quantiles to compute (controls granularity and Smoothness of Data)

## C. One Hot Encoder
One Hot Encoding is a technique to convert categorical variables (like colors, labels) into a binary numeric format suitable for machine learning models.

### Why use One Hot Encoding?
- Many ML algorithms require numerical input.

- It prevents the model from assuming any order or priority in categorical values.

- Converts each category into a separate binary (0/1) feature.

### Example
```
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample categorical data
data = np.array([['red'], ['blue'], ['green'], ['blue']])

# Create OneHotEncoder object
encoder = OneHotEncoder(sparse=False)  # sparse=False returns a dense array

# Fit and transform the data
encoded_data = encoder.fit_transform(data)

print(encoded_data)
```
### Output
```
[[0. 0. 1.]   # 'red'
 [1. 0. 0.]   # 'blue'
 [0. 1. 0.]   # 'green'
 [1. 0. 0.]]  # 'blue'
```
## D. Polynomial Features
Polynomial Features allow machine learning models to capture non-linear relationships by adding polynomial terms and interaction terms to the original features.
Ex:
 $$
x^2,\ x^3,\ x_1.x_2$$ 

```
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Sample input with two features
X = np.array([[2, 3]])

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(X_poly)
```
### Output:
[[2. 3. 4. 6. 9.]]

# Pipeline in SciKit Learn
A Pipeline in scikit-learn lets you chain multiple processing steps together (like preprocessing + modeling) into one streamlined object.

## Why Use a Pipeline?
- Clean and readable code
- Prevents data leakage

- Easily reproduce and tune the entire process

- Useful with cross-validation and grid search
```
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        ('step1_name', transformer1),
        ('step2_name', transformer2),
        # ...
        ('model_name', estimator)
    ],
    memory=None,          # Optional: caching (e.g., for grid search)
    verbose=False         # Optional: set True to show logs during fitting
)

```
Example:
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create a pipeline with scaling and logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression()) #classifier is a model that assigns categories (classes) to data points
])

# Example data
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]

# Fit the pipeline
pipe.fit(X, y)

# Predict
predictions = pipe.predict([[2, 3]])
print(predictions)
```

The pipeline guarantees that whenever you call fit or predict, it will automatically:

1. Scale the data first

2. Then train (or predict) with the logistic regression

This makes the code cleaner and less error-prone.

#  GridSearchCV
`GridSearchCV` stands for Grid Search with Cross-Validation.
GridSearchCV is a tool from scikit-learn that helps you find the best settings (hyperparameters) for your machine learning model by automatically testing all combinations you specify and evaluating each one using cross-validation

It helps you:

- Hyperparameters are settings you choose before training (like C or kernel in SVM) that can significantly affect model performance.

- Instead of guessing which settings work best, GridSearchCV tries all combinations you provide and tells you which ones perform best

## How It Works
- Define Your Model: Choose the algorithm you want to use (e.g., SVC for Support Vector Classification).

- Set Up a Parameter Grid: Make a dictionary listing the hyperparameters and all the values you want to try for each (e.g., {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}).

- Cross-Validation: For each combination of parameters, GridSearchCV splits your training data into several parts (folds), trains on some, and tests on the rest. This helps ensure the results are reliable and not just due to chance.

- Scoring: It evaluates each combination using a metric you choose, like accuracy.

- Finds the Best: After trying all combinations, it tells you which settings gave the best average performance across all folds

``` 
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,            # model (e.g., SVC(), RandomForestClassifier(), etc.)
    param_grid,           # dict or list of dicts of hyperparameters
    scoring=None,         # metric to optimize (e.g., 'accuracy', 'neg_mean_squared_error')
    n_jobs=None,          # number of jobs to run in parallel (-1 = use all CPUs)
    iid='deprecated',     # ignored (was used in older versions)
    refit=True,           # refit the best model on the full dataset
    cv=None,              # number of cross-validation folds or CV splitter (e.g., 5 or StratifiedKFold)
    verbose=0,            # control output (0 = silent, higher = more output)
    pre_dispatch='2*n_jobs', # controls how many jobs are dispatched during parallel execution
    error_score=np.nan,   # value to assign if an error occurs in fitting
    return_train_score=False # include training scores in cv_results_
)
```

- `grid.best_params_`: Best hyperparameter combination

- `grid.best_estimator_`: Best model with those parameters

- `grid.best_score_`: Best cross-validation score

- `grid.cv_results_`: Full result dictionary


```
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Model
model = SVC()

# Parameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Grid search with 5-fold cross-validation
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit on data
grid.fit(X_train, y_train)

# Best model
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
```
### Output
```
Best parameters: {'C': 1, 'kernel': 'rbf'}
Best score: 0.92
```