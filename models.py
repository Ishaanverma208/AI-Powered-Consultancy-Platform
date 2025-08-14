from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class StartCaseReq(BaseModel):
    company_name: str
    industry: str = ""
    region: str = ""
    size: str = ""
    revenue: str = ""
    users: str = ""
    problem_statement: str
    llm: str = "gemini"   # "gemini" | "groq" | "ollama"

class FollowupAnswers(BaseModel):
    case_id: str
    answers: Dict[str, str]
    llm: str = "gemini"   # same choices
