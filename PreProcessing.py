"""
Use Befor tekst_Cleaner
"""

import pandas as pd
from datetime import datetime
import locale

# Function to convert different date formats to a common format


def convert_to_common_date(date_str):
    try:
        # Set the Dutch locale for date parsing
        locale.setlocale(locale.LC_TIME, "nl_NL.utf8")

        # Try to parse the date using different date formats with Dutch locale
        date = datetime.strptime(date_str, "%A %d %B %Y, %H:%M")
    except ValueError:
        try:
            date = datetime.strptime(date_str, "%d %b %Y om %H:%M")
        except ValueError:
            try:
                date = datetime.strptime(date_str, "%d %B %Y %H:%M")
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%A %d %B")
                except ValueError:
                    try:
                        date = datetime.strptime(date_str, "%A %d %B, %H:%M")
                    except ValueError:
                        try:
                            date = datetime.strptime(date_str, "%d-%m-%Y, %H:%M")
                        except ValueError:
                            try:
                                date = datetime.strptime(date_str, "%d/%m/%Y")
                            except ValueError:
                                # If none of the formats match, return None
                                return None
    finally:
        # Reset the locale to the default
        locale.setlocale(locale.LC_TIME, "")

    return date.strftime("%d/%m/%Y")


# List of CSV files to merge
csv_files = ["Data/articles_NOS.csv", "Data/articles_NU.csv", "Data/articles_RTL.csv"]

# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Loop through the CSV files and append them to the merged DataFrame
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    merged_df = merged_df._append(df, ignore_index=True)

# Remove rows with N/A values
merged_df = merged_df.dropna()

# Apply the date format conversion function to the 'date' column
merged_df["Publication Date"] = merged_df["Publication Date"].apply(
    convert_to_common_date
)

# Filter the DataFrame to keep only records with dates on or after
# January 1, 2000
merged_df["Publication Date"] = pd.to_datetime(
    merged_df["Publication Date"], format="%d/%m/%Y"
)
merged_df = merged_df[merged_df["Publication Date"] >= "2015-01-06"]

# Convert the 'Publication Date' column back to the "%d/%m/%Y" format
merged_df["Publication Date"] = merged_df["Publication Date"].dt.strftime("%d/%m/%Y")

# Specify the name of the output merged CSV file
output_csv_file = "baselineData_PreProcessed.csv"

# Save the filtered DataFrame (with N/A values removed, common date format,
# and dates >= 2000)
# to the output CSV file
merged_df.to_csv(output_csv_file, index=False)

print(
    f"Merged CSV files with N/A values removed, common date format (Dutch), "
    f"and records on or after 01-01-2000 saved to {output_csv_file}"
)
