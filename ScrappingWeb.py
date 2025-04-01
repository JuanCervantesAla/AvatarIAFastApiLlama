import requests
from bs4 import BeautifulSoup
import json

#Urls
# https://www.biography.com/musicians/a62488913/sabrina-carpenter
# https://en.wikipedia.org/wiki/Sabrina_Carpenter

url = "https://www.biography.com/musicians/a62488913/sabrina-carpenter"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

quotes=[]

for paragraph in soup.find_all("p"): 
    text = paragraph.get_text().strip()
    if "Sabrina" in text: #Filters the quotes to Sabrina
        quotes.append({"text": text})

with open("sabrinaDataset2.json", "a", encoding="utf-8") as f:
    for quote in quotes:
        json.dump(quote,f)
        f.write("\n")