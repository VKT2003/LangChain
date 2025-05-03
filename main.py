from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

app = FastAPI()

# Allow frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Project info (hardcoded knowledge base)
project_text = """
Engineer Astra is a platform created by Vishal Tiwari that helps students get admission to top engineering colleges in India.
It provides AI-powered admission guidance, college reviews, rank predictors, student helpdesk, and free study resources like notes and lectures.
The goal is to simplify the engineering college selection and admission process for students.
"""

# 🔹 Prepare vector store from project text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents([Document(page_content=project_text)])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔹 Groq LLaMA 3 as LLM
llm = ChatOpenAI(
    model="llama3-8b-8192",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY")  # Set your key in env
)

# 🔹 LangChain QA chain
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory
)

# ✅ Routes

@app.get("/")
def root():
    return {"message": "Engineer Astra chatbot is running!"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        answer = chat_chain.run(question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
