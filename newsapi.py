import pandas as pd
import spacy
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
words_to_check = set(word_list_df["Keyword"])

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

# Filter the data based on the presence of words in the "Title" column
filtered_data_df = data_df[
    data_df["Title"]
    .str.split()
    .apply(lambda x: any(word in x for word in words_to_check))
]

# Save the filtered data to a new CSV file
filtered_data_file = "Data_final.csv"
filtered_data_df.to_csv(filtered_data_file, index=False)

print(f"Filtered data saved to {filtered_data_file}")
