from fastapi import FastAPI
from pydantic import BaseModel
from service import process

app = FastAPI()


class ErrorHandleModel(BaseModel):
    log: str

@app.post('/error')
def handle(error: ErrorHandleModel):
    return process(error.log)

@app.post('/error-simple')
def handle(error: ErrorHandleModel):
    return process(error.log)['label']