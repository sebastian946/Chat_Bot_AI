from fastapi import FastAPI

app = FastAPI(title="Chat Bot AI")


@app.get("/health")
def health():
    return {"status": "ok"}
