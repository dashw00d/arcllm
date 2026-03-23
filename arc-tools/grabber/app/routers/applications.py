"""
Application generator router.

POST /api/applications/generate  — generate personalized cover letter + tailored resume
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.recipes.application_generator import generate_application

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/applications", tags=["applications"])


class GenerateRequest(BaseModel):
    job_posting: str = Field(..., description="Full job posting text")
    company_name: str = Field(..., description="Company name for dossier lookup")
    requirements: list[str] = Field(..., description="Key requirements from the posting")
    project_id: Optional[str] = Field(None, description="Grabber project ID for dossier")
    topk: int = Field(5, description="Evidence items per requirement")


class GenerateResponse(BaseModel):
    cover_letter: str
    resume_markdown: str
    dossier_summary: str
    evidence_summary: str
    confidence: float
    generation_time_s: float
    warnings: list[str]


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate a hyper-personalized cover letter and tailored resume."""
    try:
        result = generate_application(
            job_posting=req.job_posting,
            company_name=req.company_name,
            requirements=req.requirements,
            project_id=req.project_id,
            topk=req.topk,
        )
        return result.to_dict()
    except Exception as e:
        logger.exception("Application generation failed")
        raise HTTPException(status_code=500, detail=str(e))
