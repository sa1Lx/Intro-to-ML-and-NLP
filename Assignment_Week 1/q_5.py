import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


good_feedbacks = [
    "Absolutely loved the product", "Great quality and fast delivery", "Very useful and reliable",
    "Excellent experience overall", "Highly recommended", "Will buy again", "Product works perfectly",
    "Happy with the service", "Impressed with the performance", "Very happy with the service"
] * 5  # 50 total

bad_feedbacks = [
    "Not happy with this", "Very disappointed", "Not worth the money", "Completely useless",
    "Worst product ever", "Would not recommend", "Stopped working in a day",
    "Is not good", "Unhappy with the service", "Extremely poor performance"
] * 5  # 50 total

texts = good_feedbacks + bad_feedbacks
labels = ['good'] * 50 + ['bad'] * 50

df = pd.DataFrame({'Text': texts, 'Label': labels})


vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['Text'])
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

sample = ["I am not satisfied with this product"]
vec = text_preprocess_vectorize(sample, vectorizer)
print("Predicted:", model.predict(vec)[0])
