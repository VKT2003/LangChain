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
    allow_origins=["*"],  # For development; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_PATH = "vectorstore.pkl"

@app.get("/")
def read_root():
    return {"message": "LangChain chatbot API is live 🎉"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
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
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        if not os.path.exists(VECTOR_PATH):
            return JSONResponse(status_code=400, content={"error": "Vectorstore not found. Please upload a PDF first."})

        with open(VECTOR_PATH, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        result = chain.run(question)
        return {"answer": result}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
