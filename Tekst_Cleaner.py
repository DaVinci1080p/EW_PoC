"""
Use After PreProcessing
"""

import pandas as pd
import re

# Function to clean text by removing or replacing unwanted characters


def clean_text(text):
    # Remove consecutive spaces with a single space
    cleaned_text = re.sub(r"\s+", " ", text)

    # Define a regular expression pattern to match unwanted characters
    pattern = r"[^A-Za-z0-9\s.,!?-](?![/:])"

    # Replace unwanted characters with a space
    cleaned_text = re.sub(pattern, " ", text)

    return cleaned_text


# Input CSV file with potentially weird characters
# (specify the correct encoding)
input_csv_file = "baselineData_PreProcessed.csv"

# Output CSV file with cleaned data
output_csv_file = "baselineData_PreProcessed_Clean.csv"

# Read the CSV file into a DataFrame with the specified encoding
df = pd.read_csv(input_csv_file, encoding="utf-8")

# Clean the text in Title of the DataFrame
df["Title"] = df["Title"].apply(lambda x: clean_text(str(x)))


# Save the cleaned DataFrame to a new CSV file
df.to_csv(output_csv_file, index=False)

print(f"Cleaned data saved to {output_csv_file}")
