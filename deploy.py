from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time
import logging

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize components
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Global variables to store embeddings and vector database
embeddings = None
vectors = None

# Embed documents once during initialization
def initialize_vector_store():
    global embeddings, vectors

    try:
        if embeddings is None and vectors is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            loader = PyPDFDirectoryLoader("./data")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:20])

            vectors = FAISS.from_documents(final_documents, embeddings)
            print("Vector store initialized successfully.")
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise  # Ensure the exception stops execution during initialization

@app.before_request
def check_initialization():
    global embeddings, vectors
    if embeddings is None or vectors is None:
        initialize_vector_store()




logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route("/", methods=["POST"])
def query_system():
    global vectors

    if vectors is None:
        logger.error("Vector store is not initialized.")
        return jsonify({"error": "Vector store is not initialized."}), 500

    try:
        data = request.json
        question = data.get("question")

        if not question:
            logger.warning("No question provided.")
            return jsonify({"error": "Question is required."}), 400

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        response_time = time.process_time() - start

        logger.info(f"Response generated in {response_time:.2f} seconds.")

        return jsonify({
            "answer": response.get("answer"),
            "response_time": response_time
        }), 200

    except Exception as e:
        logger.exception("Error during query processing:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
