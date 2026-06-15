import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger

from api.exceptions import generic_exception_handler, validation_exception_handler
from api.model_loader import model_loader
from api.schemas import CreditApplicationInput, ModelInfo, ScoreResponse
from ml.predict import score

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", serialize=True, level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from MLflow Registry...")
    model_loader.load()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="CreditLens Scoring API",
    description="Real-time credit risk scoring with SHAP explanations",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/api/v1/model/info", response_model=ModelInfo, tags=["model"])
def model_info():
    return ModelInfo(**model_loader.version_info)


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["scoring"])
def score_application(application: CreditApplicationInput):
    logger.info(f"Scoring application age={application.age}")
    result = score(application.model_dump(), model_loader.model, model_loader.explainer)
    return ScoreResponse(**result)
