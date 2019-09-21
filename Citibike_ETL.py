# Step 01: Get a list of the files
from bs4 import BeautifulSoup
import requests

page = requests.get('https://s3.amazonaws.com/tripdata/index.html')

# page.txt contain the entire html string
# page.text
html_doc = page.text

soup = BeautifulSoup(html_doc, 'html.parser')

# To print the formatted html 
# print(soup.prettify())

links = soup.find_all("a")

soup.prettify()