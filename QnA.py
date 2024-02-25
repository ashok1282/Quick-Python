import os
import json
from openai import OpenAI
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from fastapi import FastAPI, HTTPException, Form
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from the .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Use the API key in your OpenAI requests
import openai

# Set your OpenAI API key
openai.api_key = openai_api_key

# Check if the OpenAI API key is provided
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")

# Initialize FastAPI application
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,  # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Function to log messages with timestamps
def log_message(message, val):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message} |||| {val}\n"
    
    with open("CustomLogs.txt", "a") as log_file:
        log_file.write(log_entry)

# Function to load and correct JSON
def load_and_correct_json(json_string):
    try:
        data = json.loads(json_string)
        print("JSON is well-formed.")
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        corrected_json_string = json_string
        return corrected_json_string

# Function to load the vector index from storage
def load_vector_index():
    storage_context = StorageContext.from_defaults(persist_dir='Storedtext')
    index = load_index_from_storage(storage_context)
    return index

# Function to query the vector index with a question
def answer_me(question, index):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response


# Function to generate a prompt for question paper
def generate_question_paper_prompt(somelist):
    course_name = somelist[0]
    num_questions = somelist[1]
    input_topics= somelist[2]
    difficulty_level=somelist[3]
    topic_list=input_topics.split(',')
    topics_and_percentages = ""

    for i in range(0,len(topic_list),2):
        topics_and_percentages += f"""topic {topic_list[i]} wiht {topic_list[i+1]} %,"""
    
    question =f"""
Please generate a question paper tailored for a {course_name} course, ensuring it meets the following criteria:

Total Questions: The paper should consist of exactly {num_questions} multiple-choice questions, each with four options labeled A, B, C, and D.
Course Details: The course name is {course_name}, and the questions should reflect a "Difficult" level of {difficulty_level}.
Topic Distribution: Questions should be based on the course's specific topics, with a focus on {topics_and_percentages}covering of the paper. Please include questions from other relevant areas of {course_name} to complete the paper, ensuring a comprehensive coverage of the subject matter.
Question and Answer Format: Each question should be clearly stated, followed by four distinct options for answers. Indicate the correct answer for each question.
The output should be structured in JSON format, as shown below, with placeholders for the actual content of the questions, the options, and the correct answers:
[
  {{
    "question_number": 1,
    "question_text": "<question_text>",
    "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"],
    "correct_answer": "<correct_option>"
  }},
  ...
  {{
    "question_number": {num_questions},
    "question_text": "<question_text>",
    "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"],
    "correct_answer": "<correct_option>"
  }}
]

"""

    print(question)
    return question



# Endpoint to generate question paper using GPT-3.5 Turbo
@app.post("/generate_question_paper/gpt")
def generate_question_paper_gpt(
    course_name: str = Form(...),
    num_questions: str = Form("20"),
    input_topics: str = Form("All Categories, 100"),
    difficulty_level: str = Form("Intermediate"),
):
    num_questions = int(num_questions)
    input_variables = [
        course_name,
        num_questions,
        input_topics,
        difficulty_level,
    ]

    
    client = OpenAI(api_key=openai_api_key)
    flag = True
    retry_count = 0
    start_time = time.time()
    questions_text = generate_question_paper_prompt(input_variables)

    #for i in range(num_questions):
    try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    #messages=[questions_text],
                    temperature=0.1,
                    max_tokens=3700,
                    messages=[{"role":"system","content":"you are an Question paper generator"},
                            {"role": "user", "content":questions_text}],
                    response_format={ "type": "json_object" }
                )
            
                response_text = response.choices[0].message.content
                print(response_text)
                print (response)
                try:
                    log_message("custom log message before json parsing : ", response_text)
                    generated_questions = json.loads(response_text)
                    log_message("custom log message After json parsing : ", generated_questions)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error decoding JSON: {e}")
                    log_message("Errorrrrr message After json parsing : ", response_text)
                    raise

                print(f"Number of retries: {retry_count}")
                print(f"Time taken: {time.time() - start_time} seconds")


                return generated_questions
        
    except Exception as e:
                print(f"An error occurred: {e}")
                log_message("Errorrrrr message After json parsing : ", e)
                retry_count += 1

# Endpoint to generate question paper using a Vector Index
@app.post("/generate_question_paper/index")
def generate_question_paper_index(
    course_name: str = Form("Context"),
    num_questions: int = Form("20"),
    input_topics: str = Form("All Categories, 100"),
    difficulty_level: str = Form("Intermediate"),
):
    input_variables = [
        course_name,
        num_questions,
        input_topics,
        difficulty_level,
    ]
   
    
    vector_index = load_vector_index()
 
    prompt = generate_question_paper_prompt(input_variables)
    while(True):
        try:
            response = answer_me(prompt, vector_index)
            response_json = "{\"response\":[" + response.response+ "]}" 
            print(response_json)
            response_text = json.loads(response_json)
            
            return response_text["response"]
        except (json.JSONDecodeError, ValueError) as e:
                print(f"Error decoding JSON: {e}")
                log_message("Errorrrrr message After json parsing : ", response_json)
                raise


        
    
    

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
