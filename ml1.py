import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
import string

# Load the dataset
books_df = pd.read_csv('tp.csv', encoding='ISO-8859-1')

# Remove unnecessary columns
books_df = books_df[['title', 'description', 'genres', 'rating']]

# Define a function to clean the text
def clean_text(text):
    if type(text) == str:
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Remove stopwords and punctuation
        stopwords_list = stopwords.words('english') + list(string.punctuation)
        cleaned_tokens = [token for token in tokens if token not in stopwords_list]
        # Join the cleaned tokens back into a string
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text
    else:
        return ''


# Apply the clean_text function to the book descriptions
books_df['cleaned_description'] = books_df['description'].apply(clean_text)

# Define a function to determine the mood based on the genres and rating
def get_mood(genres, rating):
    if pd.isna(genres) or genres == '':
        return 'Unknown'
    # If the book is highly rated and in the "Mystery/Thriller" genre, return "Suspenseful"
    if rating >= 4 and 'Mystery' in genres:
        return 'Suspenseful'
    elif rating >= 4 and 'Thriller' in genres:
        return 'Suspenseful'
    # If the book is highly rated and in the "Romance" genre, return "Romantic"
    elif rating >= 4 and 'Romance' in genres:
        return 'Romantic'
    # If the book is highly rated and in the "Horror" genre, return "Scary"
    elif rating >= 4 and 'Horror' in genres:
        return 'Scary'
    # If the book is highly rated and in the "Fiction" genre, return "Happy"
    if rating >= 4 and 'Fiction' in genres:
        return 'Happy'
    # If the book is highly rated but doesn't fall into any of the above genres, return "Uplifting"
    elif rating >= 4.5:
        return 'Uplifting'
    # If the book is poorly rated and in the "Fiction" genre, return "Boring"
    elif rating < 3 and 'Fiction' in genres:
        return 'Boring'
    # If the book is poorly rated and in the "Mystery/Thriller" genre, return "Predictable"
    elif rating < 3 and 'Mystery' in genres:
        return 'Predictable'
    elif rating < 3 and 'Thriller' in genres:
        return 'Predictable'
    # If the book is poorly rated and in the "Romance" genre, return "Clichéd"
    elif rating < 3 and 'Romance' in genres:
        return 'Clichéd'
    # If the book is poorly rated and in the "Horror" genre, return "Gory"
    elif rating < 3 and 'Horror' in genres:
        return 'Gory'
    # If the book is poorly rated but doesn't fall into any of the above genres, return "Depressing"
    elif rating < 3:
        return 'Depressing'
    # If the book falls into none of the above categories, return "Neutral"
    else:
        return 'Neutral'

# Apply the get_mood function to the dataframe
print(books_df.columns)
# Apply the get_mood function to the dataframe, but only for rows where 'genres' is not null
books_df = books_df[~books_df['genres'].isnull()]
books_df['mood'] = books_df.apply(lambda row: get_mood(row['genres'], row['rating']), axis=1)
books_df.to_csv('books_with_moods1.csv', index=False)

                                  
