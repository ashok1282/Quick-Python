import os
import json
from openai import OpenAI
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from fastapi import FastAPI, HTTPException, Form
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time
import datetime

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

# Function to save response to a JSON file
def save_response_to_json(response_text, file_path='generated_response.json'):
    with open(file_path, 'w') as json_file:
        json.dump({"response": response_text}, json_file)
    
    print(f"Response saved to {file_path}")

# Function to get user input
def get_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input:
            return user_input

# Function to generate a prompt for question paper
def generate_question_paper_prompt(somelist):
    course_name = somelist[0]
    num_questions = somelist[1]
    input_topics= somelist[2]
    difficulty_level=somelist[3]
  
   
    topic_list=input_topics.split(',')
   
    topics_and_percentages={topic_list[i]: int(topic_list[i+1]) for i in range(0,len(topic_list),2)}
    
    question = f"""
    Exam Paper Generation:
 
Generate multiple-choice questions for the {course_name} course with the following specifications:
 
Number of questions: {num_questions}
Topics and Percentages:
{topics_and_percentages}
Difficulty level: {difficulty_level}
 
Please provide each question in the following JSON format, including code if required. Include four answer choices in the options section and specify the correct answer for each question. Ensure that the generated questions do not match the examples provided. Here's the expected format:
 
{{
    "question_number": 1,
    "question": "Your custom question here?",
    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    "correct_answer": "A"
}}

 
Generate the exact number of questions requested in the provided Number of questions parameter. Provide the response in JSON format only.

    """
    print(question)
    return question

# Function to make an API call to chatGpt
'''def chatGptApiCall(question, client):
    response = client.completions.create(
                model="gpt-3.5-turbo-1106",
                prompt=question,
                temperature=1,
                max_tokens=3700,
            )

    response_text = response.choices[0].text.strip()
    return response_text'''

# Endpoint to generate question paper using GPT-3.5 Turbo
@app.post("/generate_question_paper/gpt")
def generate_question_paper_gpt(
    course_name: str = Form(...),
    num_questions: str = Form(...),
    input_topics: str = Form(...),
    difficulty_level: str = Form(...),
):
    num_questions = int(num_questions)
    input_variables = [
        course_name,
        num_questions,
        input_topics,
        difficulty_level,
    ]

    questions_text = generate_question_paper_prompt(input_variables)
    print(questions_text)
    #user_content = json.dumps({"role": "user", "content": questions_text})
    #question_text=f""""role":"user","content":"give 30 c# multiple choice questions with answers and with question numbers and A,B,C,D options in json format"""
    #question =[{question_text}]
              #{"role":"user","content":"give 30 c# multiple choice questions with answers and with question numbers and A,B,C,D options in json format"}]
                           #"give 30 c# multiple choice questions with answers and with question numbers and A,B,C,D options in json format"}]
    client = OpenAI(api_key=openai_api_key)
    flag = True
    retry_count = 0
    start_time = time.time()

    for i in range(num_questions):
        try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    #messages=[questions_text],
                    temperature=0.7,
                    max_tokens=3700,
                    messages=[{"role":"system","content":"you are an Question paper generator"},
                            {"role": "user", "content":questions_text}],
                    response_format={ "type": "json_object" }
                )
            
                response_text = response.choices[0].message.content
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(response_text)
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


                return generated_questions["questions"]
        
        except Exception as e:
                print(f"An error occurred: {e}")
                log_message("Errorrrrr message After json parsing : ", e)
                retry_count += 1

# Endpoint to generate question paper using a Vector Index
@app.post("/generate_question_paper/index")
def generate_question_paper_index(
    course_name: str = Form(...),
    num_questions: int = Form(...),
    input_topics: str = Form(...),
    difficulty_level: str = Form(...),
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
            print("ssssssssssssssssssssss", response_json)
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
 question =f"""
Please generate a question paper consisting of exactly {num_questions} multiple-choice questions with options labeled as ABCD, along with their corresponding answers. 
The course for this paper is "{course_name}" and the difficulty level is "{difficulty_level}". 
Additionally, ensure that the questions are distributed across the specified topics according to the provided percentages:"{topics_and_percentages}"

JSON format for the question paper:
[
    {{"question_number": 1, "question_text": "<question_text>", "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"], "correct_answer": "<correct_option>"}},
    {{"question_number": 2, "question_text": "<question_text>", "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"], "correct_answer": "<correct_option>"}},
    {{"question_number": 3, "question_text": "<question_text>", "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"], "correct_answer": "<correct_option>"}},
    ...
    {{"question_number": {num_questions}, "question_text": "<question_text>", "options": ["A. <option1>", "B. <option2>", "C. <option3>", "D. <option4>"], "correct_answer": "<correct_option>"}}
]
"""