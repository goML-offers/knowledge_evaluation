from fastapi import FastAPI, File, UploadFile
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
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

# Initialize DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()


# Load the T5 question generation model and tokenizer
question_model = T5ForConditionalGeneration.from_pretrained("allenai/t5-small-squad2-question-generation")
question_tokenizer = T5Tokenizer.from_pretrained("allenai/t5-small-squad2-question-generation")



#model=sentence_similarity(model_name='distilbert-base-uncased',embedding_type='cls_token_embedding')
simi_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class QAChatbot:
    def __init__(self):
        self.chunks = []
        self.question_index = 0
        self.questions = []
        self.answers = []
        
    def generate_question(self, chunk_text):
    # Generate a question from the current chunk
        input_text = f"Generate a question from the following text and skip lexion plan:  {chunk_text} s"
        print(chunk_text)
        generated_question = question_model.generate(
            question_tokenizer.encode(input_text, return_tensors="pt"),
            max_length=64,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95
        )
        generated_question = question_tokenizer.decode(generated_question[0], skip_special_tokens=True)
        self.questions.append(generated_question)
        return self.questions
    

    def generate_answer(self):
        llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

        embeddings = OpenAIEmbeddings(client='')

        # Set Pinecone index
        docsearch = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        # Create chain
        chain = load_qa_chain(llm_chat)

        search = docsearch.similarity_search(chatbot.questions[-1])
        response = chain.run(input_documents=search, question=chatbot.questions[-1])
        
        print('Response:', response)
        return response

chatbot = QAChatbot()

@app.post("/upload/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
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
        # Generate a question
        query_text = chatbot.chunks[chatbot.question_index].page_content
        generated_question = chatbot.generate_question(query_text)

        # Generate an answer
        
        docs = chatbot.generate_answer()

        response_data = {
            "generated_question": generated_question[-1],
            "generated_answer": docs
        }

        return response_data
    else:
        return "No more chunks to process."
@app.post("/user_answer/")
async def user_answer(answer: str):
    generated_answer=await generate_question_and_answer()
    gen_answer=generated_answer['generated_answer']
    # Calculate TF-IDF vectors for the generated answer and the user's answer
    tfidf_matrix_generated = tfidf_vectorizer.fit_transform([gen_answer])
    tfidf_matrix_user = tfidf_vectorizer.transform([answer])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix_generated, tfidf_matrix_user)[0][0]

    if similarity > 0.8:
        responses = [
            "Outstanding! Your answer is perfect. Let's move to the next question.",
            "Fantastic! You nailed it. Next question awaits!",
            "Incredible! You're acing these questions. Let's continue.",
        ]
        response = random.choice(responses)
    elif similarity > 0.6:
        responses = [
            "Good job! You are right on the point. Let's move to the next question.",
            "Impressive! Keep up the great work. your Next question.",
            "Well done! Your understanding is shining through. On to the next one!",
        ]
        response = random.choice(responses)
    elif similarity > 0.4:
        responses = [
            "You are very close to the answer. Try answering again.",
            "Almost there! Give it another shot.",
            "Your effort is commendable, but let's try again for a more accurate answer.",
        ]
        response = random.choice(responses)
        return {"response":response}
    else:
        responses = [
            "Your effort is valued, and improvement is encouraged.Try answering again.",
            "The answer doesn't seem correct to me. Let's try one more time.",
            "Nice try! Let's see if we can get a better answer together.",
        ]
        response = random.choice(responses)
        return {"response":response}
    chatbot.question_index += 1
    return {"generated_answer": gen_answer, "user_answer": answer, "similarity": similarity,"response":response}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
