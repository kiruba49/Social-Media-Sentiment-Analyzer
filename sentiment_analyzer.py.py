import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle

# --- 1. Data Loading and Preprocessing Setup ---
# nltk.download('punkt') # Run these if you get errors!
# nltk.download('stopwords')
# nltk.download('wordnet')

data = {
    'text': [
        "This product is absolutely amazing and worth every penny!",
        "Customer service was terrible and the delivery was slow.",
        "The package arrived on time, no complaints.",
        "I hate this app; it crashes every time I open it.",
        "Great experience! Will definitely buy again.",
        "It's just okay, nothing spectacular or bad.",
        "Extremely disappointed with the quality.",
        "Best purchase of the year, highly recommend.",
        "The interface is confusing and difficult to navigate.",
        "Standard item, functions as expected."
    ],
    'sentiment': [1, 0, 2, 0, 1, 2, 0, 1, 0, 2] # 1=Positive, 0=Negative, 2=Neutral
}
df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    processed_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(processed_words)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# --- 2. Feature Engineering and Splitting ---
X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# --- 3. Model Training and Saving ---
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Saving Model and Vectorizer (CRITICAL STEP FOR GITHUB PROJECT)
# This allows others to load the trained model without running the training step
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# --- 4. Evaluation and Visualization (Output) ---
y_pred = model.predict(X_test_vectorized)
print("--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive', 'Neutral'], 
            yticklabels=['Negative', 'Positive', 'Neutral'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Word Cloud
positive_texts = " ".join(df[df['sentiment'] == 1]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words in Positive Sentiment')
plt.show()
