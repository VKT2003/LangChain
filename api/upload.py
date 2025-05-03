from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, shutil
import pickle
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open("/tmp/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    return {"message": "PDF uploaded and vectorstore created"}
