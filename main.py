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
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd

from sklearn.decomposition import PCA

app = FastAPI()

# Initialize Pinecone index
pinecone.init(
    api_key='f08932f6-f845-4fe1-bb73-ecb448ebfa0c',
    environment='gcp-starter',
)
index = pinecone.Index(index_name='knowalla')

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
    

    def get_embeddings(self, ques):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load the model from huggingface model hub
        model = SentenceTransformer("average_word_embeddings_glove.6B.300d", device=device)
        query_vectors = model.encode(ques).tolist()
        return query_vectors
    
    def reshape_vectors(self, vectors, target_dimension=300):
        reshaped_vectors = []

        for vector in vectors:
            if isinstance(vector, float):
                vector_str = str(vector)
                result_len = len(vector_str) - 1
            else:
                result_len = len(vector) - 1

            if result_len < target_dimension:
                # If the vector has fewer dimensions, pad with zeros
                padded_vector = vector + [0] * (target_dimension - result_len)
                reshaped_vectors.append(padded_vector[:target_dimension])
            elif len(vector) > target_dimension:
                # If the vector has more dimensions, truncate
                reshaped_vectors.append(vector[:target_dimension])
            else:
                reshaped_vectors.append(vector)

        return reshaped_vectors
        
    def generate_answer(self, ques):
        # Convert the generated question to embeddings
        query_vectors = self.get_embeddings(chatbot.questions[-1])
        print("vectors",query_vectors)

        target_dimension = 300
        reshaped_vectors = self.reshape_vectors(query_vectors, target_dimension)

        # Perform similarity search with Pinecone
        # result = index.query(vectors=[embeddings],top_k=3)  # Use 'vectors' parameter here
        query_results = [index.query(xq, top_k=5) for xq in reshaped_vectors]

        print("results",query_results)
        df_result=[]
        for question, res in zip(ques, query_results):
            print("\n\n\n Original question : " + str(question))
            print("\n Most similar questions based on pinecone vector search: \n")

            ids = [match.id for match in res.matches]
            scores = [match.score for match in res.matches]
            df_result.append(pd.DataFrame(
            {
                "id": ids,
                "question": ques,
                "score": scores,
            }
        ))
            print(df_result)
        # Get the most similar vectors and their indices
        # most_similar_indices = [res.id[0] for res in result]

        # Find the corresponding answers based on the most similar vectors
        # corresponding_answers = [chatbot.chunks[idx].page_content for idx in most_similar_indices]
        
        # return corresponding_answers
    
        return df_result

chatbot = QAChatbot()

@app.post("/LLM_marketplace/upload/")
async def upload_pdf(pdf_file: UploadFile):
    with open(pdf_file.filename, "wb") as f:
        f.write(pdf_file.file.read())

    loader = PyPDFLoader(pdf_file.filename)
    pages = loader.load_and_split()
    pdf_text = " ".join([page.page_content for page in pages])

    # Create document chunks
    texts = text_splitter.split_documents(pages)
    chatbot.chunks = text_splitter.create_documents([pdf_text])

    dense_vectors = []

    for text in [t.page_content for t in texts]:
        embeddings = chatbot.get_embeddings(text)
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
        # query_text = chatbot.chunks[chatbot.question_index].page_content

        generated_answer = chatbot.generate_answer(chatbot.questions[-1])
        return generated_answer
    else:
        return "No more chunks to process."

@app.post("/LLM_marketplace/user_answer/")
async def user_answer(answer: str):
    generated_answer = await generate_answer()

    # Calculate TF-IDF vector for the generated answer
    tfidf_matrix = tfidf_vectorizer.transform([generated_answer])
    generated_answer_vector = tfidf_matrix[0].tolist()

    # Perform similarity search with Pinecone
    result = index.query(vectors=[generated_answer_vector], top_k=1)

    # Get the most similar vector and its similarity score
    most_similar_vector = result[0].ids[0]
    similarity = result[0].distances[0]

    chatbot.question_index += 1
    return {"generated_answer": generated_answer, "user_answer": answer, "similarity": similarity}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
