from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from mangum import Mangum
import pickle, os

app = FastAPI()
handler = Mangum(app)

#os.environ["OPENAI_API_KEY"] = "your-groq-api-key"

@app.post("/api/ask")
async def ask(question: str = Form(...)):
    if not os.path.exists("/tmp/vectorstore.pkl"):
        return JSONResponse(status_code=400, content={"error": "No vectorstore found. Upload a PDF first."})

    with open("/tmp/vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)

    llm = ChatOpenAI(
        model="llama3-8b-8192",
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory)

    answer = qa_chain.run(question)
    return {"answer": answer}
