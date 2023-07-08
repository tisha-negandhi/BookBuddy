import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random

# Step 1: Load dataset
books_df = pd.read_csv('books_with_moods1.csv')

# Step 2: Preprocess text data
books_df['description'] = books_df['description'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['description'])

# Step 3: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, books_df['mood'], test_size=0.2, random_state=42)

# Step 4: Train mood classification model on training set
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Step 5: Evaluate model on test set
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, average="weighted", zero_division=1)}')
print(f'Recall: {recall_score(y_test, y_pred, average="weighted", zero_division=1)}')
print(f'F1 score: {f1_score(y_test, y_pred, average="weighted", zero_division=1)}')


# Step 6: Get user input
user_mood = input('How are you feeling today? ')

# Step 7: Recommend books based on mood

user_tfidf = tfidf.transform([user_mood])
user_pred = clf.predict(user_tfidf)[0]
user_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
books_df['similarity'] = user_similarities
recommended_books = books_df.loc[books_df['mood'] == user_pred].sort_values(by=['similarity'], ascending=False).head(20)
print(f'Here are the top 10 books recommended for {user_mood} mood:')
print(recommended_books[['title']])
if len(recommended_books) == 0:
    print(f'Sorry, there are no books recommended for {user_mood} mood.')
else:
    # Choose a random book from the recommended books
    random_book = recommended_books.sample(n=1)

    # Print the title and author of the random book
    print(f'Here is a random book recommended for {user_mood} mood:')
    print(f'Title: {random_book["title"].iloc[0]}')
    
