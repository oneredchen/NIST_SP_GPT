import os
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import shutil

# The URL you want to scrape from
url = "https://csrc.nist.gov/publications/search?sortBy-lg=Number+DESC&viewMode-lg=brief&ipp-lg=ALL&status-lg=Final&topicsMatch-lg=ANY&controlsMatch-lg=ANY&series-lg=SP"

print("Retrieving HTTP page...")

# Create a session object
s = requests.Session()

# Send a GET request to the URL
response = s.get(url, timeout=60)

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

print("Page retrieved. Parsing page...")

# Create a directory named 'data' to save the downloaded files
if not os.path.exists("data"):
    os.makedirs("data")

# Find the elements with id that starts with 'download-value-' and ends with '-1'
elements = soup.find_all(
    id=lambda x: x and x.startswith("download-value-") and x.endswith("-1")
)

print("Page parsed. Found {} files to download.".format(len(elements)))


def download_file(element):
    # Get the file name from the download link
    download_link = element.find("a")["href"]
    file_name = download_link.split("/")[-1]
    file_path = os.path.join("data", file_name)

    try:
        # Check if file already exists
        if os.path.exists(file_path):
            print("File {} already exists. Skipping download.".format(file_name))
            return

        print("Downloading file from {}...".format(download_link))

        # Send a GET request to the download link
        file_response = s.get(download_link, stream=True)

        # Save the file to the 'data' directory
        with open(file_path, "wb") as file:
            shutil.copyfileobj(file_response.raw, file)

        print("File {} downloaded.".format(file_name))
    except Exception as e:
        print("Failed to download file. Error: {}".format(e))

        # If an error occurred during the download, delete the file
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Incomplete file {} removed.".format(file_name))


# Use ThreadPoolExecutor todownload multiple files concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(download_file, elements)

print("Download completed.")
