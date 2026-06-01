import pandas as pd
import collections
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("spam_sms.csv")
print("Dataset shape:", data.shape)  # (5572, 2)

# Rename columns
data = data.rename(columns={'v1': "spam", 'v2': "message"})

# Encode labels (ham = False, spam = True)
data['spam'] = data['spam'].map({'ham': False, 'spam': True})

# Check class distribution
print("\nSpam (True) and Ham (False) counts:\n", data['spam'].value_counts())

# Function to count word frequencies
def count_words(messages):
    counter = collections.OrderedDict()
    for message in messages:
        for word in message.split():
            word = word.lower().strip()
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter

# Function to plot top words
def plot_top_words(word_counts, title, top_n=10):
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='orange')
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Get top spam and ham words and plot
spam_messages = data[data['spam'] == True]['message']
ham_messages = data[data['spam'] == False]['message']

spam_words = count_words(spam_messages)
ham_words = count_words(ham_messages)

plot_top_words(spam_words, "Top Words in Spam Messages")
plot_top_words(ham_words, "Top Words in Ham Messages")

# Feature extraction using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])  # Sparse matrix
y = data['spam']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_model = BernoulliNB()
nb_model.fit(X_train, y_train)

# Predict
y_pred = nb_model.predict(X_test)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
# print("\n Classification Report:\n", classification_report(y_test, y_pred))
# print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
