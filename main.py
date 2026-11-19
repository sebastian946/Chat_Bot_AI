from fastapi import FastAPI

from db.postgress import lifespan
from routes.routes import router

# lifespan creates the tables when the app starts, so Postgres must be running.
app = FastAPI(title="Chat Bot AI", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
