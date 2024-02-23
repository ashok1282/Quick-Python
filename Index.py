from fastapi import FastAPI
import os
import fitz
import docx
import textract
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Use the API key in your OpenAI requests
# Example: print(openai_api_key)
import openai

# Set your OpenAI API key
openai.api_key = openai_api_key
app = FastAPI()

combined_file_path = "Newcombinedpdfs/combined.txt"  
folder_path = "docs"
from pathlib import Path
from datetime import datetime

def store_data(directory, username, subject, data):
    """
    Stores data in a new file within a structured directory path. Each file is uniquely named using a timestamp.

    :param directory: Base directory path
    :param username: User's name or identifier
    :param subject: Subject or category of the data
    :param data: The data to be stored
    """
    # Construct the path
    path = Path(directory) / username / subject
    
    # Ensure the directory exists
    path.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}.txt"
    
    # Define the file path
    file_path = path / file_name
    
    # Write data to the new file
    with file_path.open("w") as file:
        file.write(data + "\n")
    
    # Return the full path of the created file
    return file_path

# Example usage
directory = '/home/tammali/Documents/QAG/QnAData'  # Make sure to adjust this path to your actual directory
username = 'john_doe'
subject = 'math'
data = 'This is a sample data entry.'

created_file_path = store_data(directory, username, subject, data)
print(f"Data stored in {created_file_path}")



def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text


def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_other(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text


@app.get("/")
def combine_and_index():
    with open(combined_file_path, "w", encoding="utf-8") as combined_file:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as txt_file:
                    txt_content = txt_file.read()
                    combined_file.write(txt_content + "\n")
            elif filename.endswith(".pdf"):
                pdf_text = extract_text_from_pdf(file_path)
                combined_file.write(pdf_text + "\n")
            elif filename.endswith(".docx"):
                docx_text = extract_text_from_docx(file_path)
                combined_file.write(docx_text + "\n")
            else:
                other_text = extract_text_from_other(file_path)
                combined_file.write(other_text + "\n")

    create_vector_index('Newcombinedpdfs')
    return {"message": "Index has been stored in the 'Storedtext' directory."}


def create_vector_index(path):
    max_input = 4096
    tokens = 4000
    chunk_size = 600
    max_chunk_overlap = 0
    prompt_var = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    llm_predictor = LLMPredictor(llm=OpenAI( temperature=1,model_name="gpt-3.5-turbo", max_tokens=tokens))

    docs = SimpleDirectoryReader(path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_var)
    vector_index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    vector_index.storage_context.persist(persist_dir='Storedtext')


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
