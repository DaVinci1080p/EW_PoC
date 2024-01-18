import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
import csv
import re
import sys

sys.setrecursionlimit(10000)
default_recursion_limit = sys.getrecursionlimit()
print("Default recursion depth limit:", default_recursion_limit)

# Define a regular expression pattern to match URLs to be skipped
skip_pattern = r"^https://www\.rtlnieuws\.nl/nieuws/video/"
skip_pattern2 = r"^https://www\.rtlnieuws\.nl/sites/default"
skip_pattern3 = r"^https://www\.rtlnieuws\.nl/video"

# Function to crawl a URL and its subpages


def crawl_url(url, visited_urls, csv_writer):
    # Normalize the URL by removing fragments
    url, _ = urldefrag(url)

    # Check if the URL has already been visited or matches the skip pattern
    if (
        url in visited_urls
        or re.match(skip_pattern, url)
        or re.match(skip_pattern2, url)
        or re.match(skip_pattern3, url)
    ):
        return

    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract the publication date (customize this based on the
            # website's structure)
            publication_date = soup.find(
                "time", class_="node__created"
            )  # Adjust as needed

            if publication_date:
                publication_date = publication_date.text.strip()
            else:
                publication_date = "N/A"

            # Extract the title of the article (customize this based on the
            #  website's structure)
            article_title = soup.find("h1", class_="article__title")  # Adjust as needed

            if article_title:
                article_title = article_title.text.strip()
            else:
                article_title = "N/A"

            # Print the URL and publication date for demonstration purposes
            print(f"URL: {url}, Publication Date: {publication_date}")

            # Write the URL, title and publication date to the CSV file
            csv_writer.writerow([url, article_title, publication_date])

            # Add the URL to the set of visited URLs
            visited_urls.add(url)

            # Find all links on the page and crawl them recursively
            for link in soup.find_all("a"):
                href = link.get("href")
                if href:
                    absolute_url = urljoin(url, href)
                    # Check if the link is within the desired path
                    if absolute_url.startswith("https://www.rtlnieuws.nl"):
                        crawl_url(absolute_url, visited_urls, csv_writer)
    except Exception as e:
        print(f"Failed to crawl URL {url}: {str(e)}")


# Start crawling with the initial URL
initial_url = "https://www.rtlnieuws.nl"
visited_urls = set()  # Initialize the set of visited URLs

# Create a CSV file and write the header
with open("articles_RTL.csv", "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["URL", "Title", "Publication Date"])

    # Call the crawl function
    crawl_url(initial_url, visited_urls, csv_writer)
