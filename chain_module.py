import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import pandas as pd

load_dotenv()

csv_path = "dataset.csv"
vectordb_file_path = "faiss_index"

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
     model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def create_vector_db():
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"{csv_path} not found!")

#     try:
#         loader = CSVLoader(file_path=csv_path, source_column="prompt")
#         documents = loader.load()
#     except Exception as e:
#         print(f"CSVLoader failed. Falling back to manual load: {e}")
#         df = pd.read_csv(csv_path)
#         documents = [
#             Document(page_content=row["prompt"], metadata={"response": row["response"]})
#             for _, row in df.iterrows()
#         ]

#     vectordb = FAISS.from_documents(documents, embedding=embeddings)
#     vectordb.save_local(vectordb_file_path)

# def create_vector_db():
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"{csv_path} not found!")

#     try:
#         loader = CSVLoader(file_path=csv_path, source_column="prompt")
#         documents = loader.load()
#     except Exception as e:
#         print(f"CSVLoader failed. Falling back to manual load: {e}")
#         df = pd.read_csv(csv_path)
#         df = df.dropna(subset=["prompt"])  # 🚨 Drop rows with missing prompts
#         documents = [
#             Document(page_content=row["prompt"], metadata={"response": row["response"]})
#             for _, row in df.iterrows()
#         ]

#     vectordb = FAISS.from_documents(documents, embedding=embeddings)
#     vectordb.save_local(vectordb_file_path)

def create_vector_db():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found!")

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["prompt", "response"])  # Drop rows with missing values

        documents = [
            Document(page_content=str(row["prompt"]), metadata={"response": str(row["response"])})
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to process dataset: {e}")

    vectordb = FAISS.from_documents(documents, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """
    You are a helpful medical assistant. Use the following context to answer the user's medical question.
    If the answer is not fully in the context, you may use your general medical knowledge, but mention that it's not directly from the MedQuAD source.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """


    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain
