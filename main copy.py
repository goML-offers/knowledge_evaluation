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



app = FastAPI()

# Initialize Pinecone index
pinecone.init(
    api_key='f08932f6-f845-4fe1-bb73-ecb448ebfa0c',
    environment='gcp-starter'
)
index = pinecone.Index('knowalla')

# Initialize text processing components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=lambda text: len(text)
)

# Initialize DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Initialize OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

OPENAI_API_KEY="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"

# Load the T5 question generation model and tokenizer
question_model = T5ForConditionalGeneration.from_pretrained("allenai/t5-small-squad2-question-generation")
question_tokenizer = T5Tokenizer.from_pretrained("allenai/t5-small-squad2-question-generation")

class QAChatbot:
    def __init__(self):
        self.chunks = []
        self.question_index = 0
        self.questions = []
        self.answers = []
        
    def generate_embeddings(self, chunk_text):
        inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings
    def generate_question(self, chunk_text):
    # Generate a question from the current chunk
        input_text = f"Generate a question from the following text: {chunk_text}"
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
        return generated_question
    

    async def generate_answer():
        if chatbot.question_index < len(chatbot.chunks):
            query_text = chatbot.chunks[chatbot.question_index].page_content

            # Retrieve the vectors from Pinecone using Langchain
    

            # Get the most similar vector and its index
            most_similar_index = result[0]['id']

            # Find the corresponding answer based on the most similar vector
            chunk_text = chatbot.chunks[most_similar_index].page_content
            return chunk_text
        else:
            return "No more chunks to process."
    
    def get_embedding(self,text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        openai.api_key="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"

        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

    

chatbot = QAChatbot()

@app.post("/LLM_marketplace/upload/")
async def upload_pdf(pdf_file: UploadFile):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())

    loader = PyPDFLoader(pdf_file.filename)
    # pages = loader.load_and_split()
    # pdf_text = " ".join([page.page_content for page in pages])
    

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # documents = text_splitter.split_documents(loader)
    # db = Chroma.from_documents(documents, OpenAIEmbeddings())


    # Create document chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(pdf_text)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts, embeddings)
    retriever = db.as_retriever()
    dense_vectors = []

    for text in [t.page_content for t in texts]:
        embeddings = chatbot.generate_embeddings(text)
        dense_vectors.append(embeddings)

    # Prepare data for upsert
    data = [{'id': str(i), 'values': vector} for i, vector in enumerate(dense_vectors)]

    # Upload vectors to Pinecone
    index.upsert(data)
    
@app.get("/LLM_marketplace/generate_question/")
async def generate_question():
    if chatbot.question_index < len(chatbot.chunks):
        # Retrieve a chunk of text from Pinecone
        # result = index.query(queries=[chatbot.chunks[chatbot.question_index].page_content], top_k=1)
        # chunk_text = result[0].result.ids[0]

        # Generate a question from the retrieved chunk
        generated_question = chatbot.generate_question(chatbot.chunks)

        return generated_question
    else:
        return "No more chunks to process."
        
@app.get("/LLM_marketplace/generate_answer/")
async def generate_answer():
    if chatbot.question_index < len(chatbot.chunks):
        query_text = chatbot.chunks[chatbot.question_index].page_content
        docs = chatbot.db.similarity_search(query_text)
        print(docs[0].page_content)
        return docs
    else:
        return "No more chunks to process."

@app.post("/LLM_marketplace/user_answer/")
async def user_answer(answer: str):
    generated_answer = await generate_answer()

    # Calculate TF-IDF vector for the generated answer
    tfidf_matrix = tfidf_vectorizer.transform([generated_answer])
    generated_answer_vector = tfidf_matrix[0].tolist()

    # Perform similarity search with Pinecone
    result = index.query(queries=[generated_answer_vector], top_k=1)

    # Get the most similar vector and its similarity score
    most_similar_vector = result[0].ids[0]
    similarity = result[0].distances[0]

    chatbot.question_index += 1
    return {"generated_answer": generated_answer, "user_answer": answer, "similarity": similarity}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
