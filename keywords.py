import csv
import nltk
import spacy
import re
from nltk.corpus import stopwords
from translate import Translator

# Download NLTK data and NLTK stopwords
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words_nl = set(stopwords.words("dutch"))

# Load the Dutch language model from spaCy
nlp = spacy.load("nl_core_news_sm")

# Create an empty set to store unique non-stopwords
unique_non_stopwords = set()

# Specify the path to your CSV file
csv_file_path = "Data/keywords.csv"  # Replace with your file path

# Open the CSV file and extract and store unique non-stopwords
with open(csv_file_path, "r", newline="") as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row if it exists

    for row in csv_reader:
        keyword = row[0]
        words = keyword.split()
        for word in words:
            # Convert the word to lowercase to ensure uniformity
            word = word.lower()
            # Check if the word is a stopword
            if word not in stop_words:
                unique_non_stopwords.add(word)

# Convert the set of unique non-stopwords back to a list
unique_non_stopwords_list = list(unique_non_stopwords)

# Print the list of unique non-stopwords
print(unique_non_stopwords_list)

print(len(unique_non_stopwords_list))

translated_words = []

# Translate each word from English to Dutch
translator = Translator(to_lang="nl")  # "nl" is the language code for Dutch

for word in unique_non_stopwords_list:
    translated_word = translator.translate(word)
    translated_words.append(translated_word)

# Create an empty list to store the filtered words
filtered_words = []

# Initialize a set to keep track of unique words
unique_words = set()

translated_words_split = []
for i in translated_words:
    if "/" in i:
        translated_words_split.extend(i.split("/"))
    else:
        translated_words_split.append(i)
print(translated_words_split)

for word in translated_words_split:
    # Remove special characters using regex
    print(word)
    word = re.sub(r"[^A-Za-z0-9\s]", "", word)
    word = word.strip().lower()
    if word not in stop_words_nl and word not in unique_words:
        filtered_words.append(word)
        unique_words.add(word)

for word in filtered_words:
    # Remove special characters using regex
    word = re.sub(r"[^A-Za-z0-9\s]", "", word)

keywords_clean = filtered_words + unique_non_stopwords_list

csv_file_path = "Data/keywords_clean.csv"
with open(csv_file_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Keyword"])  # Write the header

    for keyword in keywords_clean:
        csv_writer.writerow([keyword])

print(f"Saved {len(keywords_clean)} keywords to {csv_file_path}")
