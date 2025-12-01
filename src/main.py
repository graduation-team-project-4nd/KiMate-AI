from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ai_service import AIService
from .models import AnalyzeRequest, AnalyzeResponse, ScreenDetectRequest, ScreenDetectResponse
from .screen_detect import ScreenDetector

load_dotenv()  # Load variables from .env if present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KiMate-AI Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _init_ai_service() -> AIService:
    model = os.getenv("OPENAI_MODEL", "gpt-5.1")
    mock = os.getenv("AI_SERVER_MOCK", "0") == "1"
    logger.info("AI service init - model=%s mock=%s", model, mock)
    return AIService(model=model, mock=mock)


def _init_screen_detector(ai_service: AIService) -> ScreenDetector:
    threshold = float(os.getenv("SCREEN_CHANGE_THRESHOLD", "0.6"))
    return ScreenDetector(ai_service=ai_service, threshold=threshold)


ai_service = _init_ai_service()
screen_detector = _init_screen_detector(ai_service)


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return await ai_service.analyze(req)


@app.post("/api/screen/detect", response_model=ScreenDetectResponse)
async def screen_detect(req: ScreenDetectRequest) -> ScreenDetectResponse:
    return await screen_detector.detect(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
