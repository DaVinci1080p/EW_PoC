import csv
import datetime

# Input and output file names
input_file = "Data/energyPrice.csv"
output_file = "EnergyPrice_Clean.csv"

# Define a function to convert a Unix timestamp to the desired format


def convert_unix_to_date(unix_timestamp):
    date_object = datetime.datetime.utcfromtimestamp(
        int(unix_timestamp) / 1000
    )  # Assuming timestamps are in milliseconds
    return date_object.strftime("%d/%m/%Y")


# Open the input and output CSV files
with open(input_file, "r") as csvfile, open(output_file, "w", newline="") as outputcsv:
    reader = csv.reader(csvfile)
    writer = csv.writer(outputcsv)

    # Iterate through rows in the input file and convert timestamps
    for row in reader:
        if row:
            unix_timestamp = row[0]
            formatted_date = convert_unix_to_date(unix_timestamp)
            new_row = [formatted_date] + row[1:]
            writer.writerow(new_row)
