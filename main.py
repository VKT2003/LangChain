from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from mangum import Mangum
import shutil
import os

app = FastAPI()

#os.environ["OPENAI_API_KEY"] = "your-groq-api-key"

llm = ChatOpenAI(
    model="llama3-8b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vectorstore = None
qa_chain = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    global vectorstore, qa_chain
    vectorstore = FAISS.from_documents(docs, embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return {"message": "PDF uploaded and processed"}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not qa_chain:
        return JSONResponse(status_code=400, content={"error": "Please upload a PDF first"})
    response = qa_chain.run(question)
    return {"answer": response}

handler = Mangum(app)
