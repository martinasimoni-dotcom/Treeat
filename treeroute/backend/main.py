from fastapi import FastAPI

app = FastAPI(title="TreeRoute API")


@app.get("/")
def root():
    return {"status": "ok"}
