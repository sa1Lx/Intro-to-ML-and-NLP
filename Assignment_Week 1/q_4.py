import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

positive_reviews = ["I loved the movie!", "Amazing film!", "Great acting!", "Wonderful story!", "Enjoyed it a lot!",
                    "Awesome direction!", "Very touching!", "Brilliant!", "Heartwarming!", "Excellent!", 
                    "Fantastic experience!", "Top class!", "Superb visuals!", "Emotional and strong!", "Simply perfect!",
                    "Highly recommended!", "Will watch again!", "Very well made!", "Classic hit!", "Stellar work!",
                    "Loved every part!", "Best movie ever!", "Super engaging!", "Oscar-worthy!", "Truly beautiful!",
                    "A masterpiece!", "Great soundtrack!", "Must watch!", "Beautifully executed!", "Inspiring!",
                    "Five stars!", "Strong script!", "Marvelous!", "Absolutely loved it!", "Well directed!",
                    "Full marks!", "Jaw dropping!", "Totally enjoyed!", "Top notch acting!", "Perfectly paced!",
                    "Amazing chemistry!", "Impressive work!", "Story was gripping!", "True gem!", "Outstanding!",
                    "Just wow!", "Unforgettable!", "Loved the characters!", "What a ride!", "Super entertaining!",
                    "Cinematography was amazing!"]

negative_reviews = ["I hated the movie.", "Terrible film.", "Bad acting.", "Horrible story.", "Didn't like it.",
                    "Poor direction.", "Very boring.", "Disappointing.", "Waste of time.", "Awful!",
                    "No emotion.", "Worst ever.", "Super dull.", "Too slow.", "What a mess!",
                    "Regret watching.", "Not enjoyable.", "Boring plot.", "Unwatchable.", "Just plain bad.",
                    "Too predictable.", "Overrated!", "Forgettable movie.", "Pathetic acting.", "Weak dialogues.",
                    "Didn't connect.", "Tired storyline.", "Zero chemistry.", "Very loud.", "Nonsense plot.",
                    "No substance.", "Terribly executed.", "Underwhelming.", "Too cliche.", "Missed potential.",
                    "Fails to impress.", "Flat performance.", "Bad pacing.", "Fake emotions.", "Too long!",
                    "Disaster.", "Lazy writing.", "Felt forced.", "Dry humor.", "Cried for wrong reason.",
                    "Had to skip.", "Turned it off.", "Wasted cast.", "Fell asleep.", "Don't recommend."]

positive_reviews = positive_reviews[:50]
negative_reviews = negative_reviews[:50] # debugging since i randomly generated and got error for unequal lengths

reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

def predict_review_sentiment(model, vectorizer, review):
    vec_review = vectorizer.transform([review])
    return model.predict(vec_review)[0]

print("\nPrediction for 'This movie was brilliant and touching!':")
print(predict_review_sentiment(model, vectorizer, "This movie was brilliant and touching!"))

print("\nPrediction for 'It was a total disaster, waste of time.':")
print(predict_review_sentiment(model, vectorizer, "It was a total disaster, waste of time."))


