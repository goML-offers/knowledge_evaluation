from fastapi import FastAPI, File, UploadFile
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Index
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
from sentence_similarity import sentence_similarity
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
import random
from dotenv import load_dotenv
import pandas as pd
import json

import re
load_dotenv()

PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')  # 'kn1'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')


# Set API key for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

print(OPENAI_API_KEY)
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)


print('Checking if index exists...')
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    print('Index does not exist, creating index...')
    # we create a new index
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        metric='cosine',
        # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
        dimension=1536
    )

print('Loading document...')

app = FastAPI()

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["GET", "OPTIONS", "POST", "PUT", "DELETE"],

    allow_headers=["*"],

)


class QAChatbot:
    def __init__(self):
        self.chunks = []
        self.question_index = 0
        self.questions = []
        self.answers = []

    def generate_question(self):
    # Generate a question from the current chunk
        # input_text = f"Generate a question from the following text and skip lexion plan:  {chunk_text} s"
        # print(chunk_text)
        # generated_question = question_model.generate(
        #     question_tokenizer.encode(input_text, return_tensors="pt"),
        #     max_length=64,
        #     num_return_sequences=1,
        #     no_repeat_ngram_size=2,
        #     top_k=50,
        #     top_p=0.95
        # )
        # generated_question = question_tokenizer.decode(generated_question[0], skip_special_tokens=True)
        # self.questions.append(generated_question)
        # return self.questions
        llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        # Set Pinecone index
        docsearch = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        # Create chain
        chain = load_qa_chain(llm_chat)

        search = docsearch.similarity_search('list down 6 follow up questions based on the documents in a json array format.For example ["question1","question2"]')
        response = chain.run(input_documents=search, question='list down 6 follow up questions based on the documents in a json array format.For example ["question1","question2"]')
        
        #print('Response:', response)
        print(response)
        print("===============")
        # data=pd.DataFrame({"response":eval(response)})
        # data.to_csv('data.csv')
        # addon=["Allow me to pose a question that enriches your learning experience. ",
        #         "Allow me to stimulate your learning with a thoughtful question. ",
        #         "Presenting a query to brush up on your learning endeavors. ",
        #         "Allow me to ask you a question that adds a sparkle to your learning. "]
        # addon = random.choice(addon)
        # if os.path.exists('data.csv'):
        #     addon+data['response'].iloc[0]
        # print(addon+data['response'].iloc[0])
        response=re.sub(r'[^a-zA-Z0-9\s,]', '', response)
        response=list(response.split(','))
        
        print(response)
        return response
    
    def generate_answer(self, current_question):
        llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        # Set Pinecone index
        docsearch = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        # Create chain
        chain = load_qa_chain(llm_chat)

        # Use the provided current question instead of the last question in the list
        search = docsearch.similarity_search(current_question)
        response = chain.run(input_documents=search, question=current_question)
        
        print('Response:', response)
        print("++=========================",search)
        search=search[0].page_content
        print(search)
        return response,search

    def ask_GPT3(self,str1,str2): 
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        model = 'gpt-3.5-turbo',
        messages = [ # Change the prompt parameter to the messages parameter
        {'role': 'user', 'content': f'what is the exact similarity between two strings in an array below : [{str1}, {str2}]. Provide a decimal number as an output with no description! for example 0.55'}
        ],
        temperature = 0  
        )
        return completion['choices'][0]['message']['content']


chatbot = QAChatbot()

@app.post("/upload/")
async def upload_pdf(pdf_file: UploadFile):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())

    loader = PyPDFLoader(pdf_file.filename)
    pages = loader.load()
    pdf_text = " ".join([page.page_content for page in pages])
    # Chunk data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)

    # Create document chunks
    texts = text_splitter.split_documents(pages)
    chatbot.chunks = text_splitter.create_documents([pdf_text])


    # print(f'You have split your document into {len(texts)} smaller documents')

    # Create embeddings and index from your documents
    # print('Creating embeddings and index...')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)
    os.remove(pdf_file.filename)
    print('Done!')

@app.post("/generate_ques_ans/")
async def generate_question_and_answer():
    if chatbot.question_index < len(chatbot.chunks):
        # Generate questions
        # query_text = chatbot.chunks[chatbot.question_index].page_content
        generated_questions = chatbot.generate_question()
        print(generated_questions)
        print(type(generated_questions))
        # Initialize an empty list to store the responses
        responses_list = []
         
        question_count=len(generated_questions)
        # Iterate through the generated questions and generate answers
        for i in range(question_count):
            # Generate an answer
            gen_ques=generated_questions[i]
            gen_answer,search= chatbot.generate_answer(gen_ques)
            # print(search)

           

            # Create a response dictionary
            response_data = {
                "question": gen_ques,
                "generated_answer": gen_answer,
                "document":search ,
            }

            # Append the response to the list
            responses_list.append(response_data)

        # # Increment the question index
        # chatbot.question_index += 1

        # Return the list of responses
        return responses_list
    else:
        return "No more chunks to process."

@app.post("/user_answer/")
async def user_answer(answer: str,system_answer:str):
    # response=await generate_question_and_answer()
    # print(response)
    # gen_answer=response[0]['generated_answer']
    similarity = chatbot.ask_GPT3(answer,system_answer)
    print(type(similarity))
    print(similarity)
    similarity=float(similarity)
    if similarity > 0.4:
        responses = [
            "Outstanding! Your answer is perfect. Let's move to the next question.",
            "Fantastic! You nailed it. Next question awaits!",
            "Incredible! You're acing these questions. Let's continue.",
        ]
        response = random.choice(responses)
    else:
        responses = [
            "OOPs! Take a moment to deepen your understanding. This is where you should be going to improve your knowledge",
            "Great effort! But you need to read more.This is where you should be going to improve your knowledge",
            "OhOoh! I think you need to brush up this concept once again to become an expert.This is where you should be going to improve your knowledge ",
            "While you are the teacher here, but I don't think this is the correct answer. Can you please brush this topic up?For now, Here is the correct answer I can acecss."
        ]
        response = random.choice(responses)
    chatbot.question_index += 1
    
    return {"generated_answer": system_answer, "user_answer": answer, "similarity": similarity,"response":response}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
