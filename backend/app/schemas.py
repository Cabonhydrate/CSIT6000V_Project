from typing import Literal
from pydantic import BaseModel, Field, model_validator
from .agent.memory_management import is_memory_listing_request


class ChatRequest(BaseModel):
    conversation_id: str
    question: str = Field(min_length=1, max_length=6000)
    paper_ids: list[str] = Field(default_factory=list, max_length=8)
    scope: Literal['paper','general'] = 'paper'
    mode: Literal['basic','hybrid','adaptive'] = 'adaptive'
    teaching: bool = True
    use_memory: bool = True
    allow_web: bool = False

    @model_validator(mode='after')
    def validate_scope(self):
        if self.scope == 'paper' and not self.paper_ids and not is_memory_listing_request(self.question):
            raise ValueError('Paper QA requires at least one selected paper')
        if self.scope == 'general':
            if self.paper_ids:
                raise ValueError('General Learning must not cite selected papers')
            if self.mode != 'adaptive':
                raise ValueError('General Learning currently uses Adaptive mode')
        return self


class ConversationCreate(BaseModel):
    title: str = Field(default='New conversation', min_length=1, max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ProfileUpdate(BaseModel):
    language: Literal['zh','en'] = 'en'
    explanation_level: Literal['beginner','intermediate','advanced'] = 'beginner'
    domains: list[str] = Field(default_factory=list, max_length=20)
    goals: list[str] = Field(default_factory=list, max_length=20)
    memory_enabled: bool = True


class MemoryCreate(BaseModel):
    domain: str = Field(min_length=1, max_length=120)
    topic: str = Field(min_length=1, max_length=120)
    memory_type: Literal['preference','goal','progress'] = 'progress'
    content: str = Field(min_length=1, max_length=1000)
    mastery: Literal['mentioned','studied','confirmed'] = 'mentioned'


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=3000)
    paper_ids: list[str] = Field(min_length=1, max_length=8)
    strategy: Literal['keyword','dense','hybrid'] = 'keyword'
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
