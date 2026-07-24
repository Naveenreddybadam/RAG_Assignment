import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Download webpage
url = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2"
response = requests.get(url)

if response.status_code != 200:
    print("Failed to download website.")
    exit()

print("Website downloaded successfully!")

# Step 2: Extract text
soup = BeautifulSoup(response.text, "html.parser")
text = soup.get_text(separator="\n", strip=True)

# Save original text
with open("data/pib_document.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("Text extracted successfully!")

# Step 3: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

# Save chunks
with open("data/chunks.txt", "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"\n========== Chunk {i+1} ==========\n")
        file.write(chunk)
        file.write("\n")

print("Chunks saved successfully!")