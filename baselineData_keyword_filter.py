import pandas as pd
import spacy
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Load the CSV file to check for words in the "Title" column
word_list_file = (
    "Data/keywords_clean.csv"  # Replace with the path to your word list CSV file
)
word_list_df = pd.read_csv(word_list_file)

# Load the CSV file to filter
data_file = "Data/baselineData_PreProcessed_Clean.csv"  # Replace with the path to your data CSV file
data_df = pd.read_csv(data_file)

# Create a set of words to check for (both Dutch and English)
words_to_check = set(
    word_list_df["Keyword"]
)  # Assuming 'Keyword' is the column with the words to check

# Initialize the spaCy language models for Dutch and English
nlp_dutch = spacy.load("nl_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")


# Define a function to lemmatize text based on the language
def lemmatize_text(text, language):
    if language == "nl":
        doc = nlp_dutch(text)
    else:
        doc = nlp_english(text)
    return " ".join([token.lemma_ for token in doc])


# Define a function to detect the language of a text
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        # If language detection fails, return None
        return None


# Detect the language of each word in the word list
word_languages = {word: detect_language(word) for word in words_to_check}

# Lemmatize the "Title" column in the data based on the detected language
data_df["Title"] = data_df.apply(
    lambda row: lemmatize_text(row["Title"], word_languages.get(row["Title"], "en")),
    axis=1,
)

# Create a regular expression pattern to match words from the word list
word_pattern = r"\b" + "|".join(map(re.escape, words_to_check)) + r"\b"


# Create a function to check if any word from the word list is in the text
def contains_word_from_list(text):
    return re.search(word_pattern, text) is not None


# Filter the data based on the presence of connected words from the word list in the "Title" column
filtered_data_df = data_df[data_df["Title"].apply(contains_word_from_list)]

# Save the filtered data to a new CSV file
filtered_data_file = (
    "filtered_data.csv"  # Replace with the desired output CSV file path
)
filtered_data_df.to_csv(filtered_data_file, index=False)

print(f"Filtered data saved to {filtered_data_file}")
