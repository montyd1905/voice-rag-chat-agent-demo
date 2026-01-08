from typing import Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    rectified_query: Optional[str] = None
    response: str
    source: str
    entities: list
    session_id: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    filename: Optional[str] = None
    chunks: Optional[int] = None
    entities: Optional[int] = None