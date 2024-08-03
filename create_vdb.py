from pypdf import PdfReader
import re

def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

# replace the path with your file path
pdf_text = load_pdf(file_path="coi.pdf")

def save_text_to_file(text, output_file_path):
    """
    Saves the given text to a text file.

    Parameters:
    - text (str): The text content to save.
    - output_file_path (str): The file path where the text will be saved.
    """
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)

save_text_to_file(text=pdf_text, output_file_path="output.txt")

from hide_it import hf_api
inference_api_key = hf_api

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

loader = TextLoader("output.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, length_function=len)
docs = text_splitter.split_documents(documents)
print(len(docs))
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

db.save_local("faiss_index_law")