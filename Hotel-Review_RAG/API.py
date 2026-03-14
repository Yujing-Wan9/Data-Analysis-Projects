from fastapi import FastAPI
from pydantic import BaseModel

# 匯入你原本的 RAG function
from hotelreview_Demo import ask_hotel

app = FastAPI(title="Hotel Review RAG API")

class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Hotel Review RAG API is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):

    answer = ask_hotel(request.question)

    return {
        "question": request.question,
        "answer": answer
    }
