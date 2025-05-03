from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pickle, os, shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_PATH = "vectorstore.pkl"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    return {"message": "Vectorstore created from PDF"}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not os.path.exists(VECTOR_PATH):
        return JSONResponse(status_code=400, content={"error": "No vectorstore found"})

    with open(VECTOR_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    llm = ChatOpenAI(
        model="llama3-8b-8192",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ.get("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory)
    answer = chain.run(question)

    return {"answer": answer}
