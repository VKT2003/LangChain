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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Static project info
project_text = """
Engineer Astra is a platform that helps students get admission to top engineering colleges in India.
It provides college reviews, rank predictors, admission guidance, AI-based helpdesk, and free notes and lectures.
Founded by Vishal Tiwari, the platform aims to simplify the college selection and admission process.
"""

# 🔹 Preprocess and embed
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents([Document(page_content=project_text)])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

@app.get("/")
def root():
    return {"message": "Engineer Astra chatbot is live!"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        answer = qa_chain.run(question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
