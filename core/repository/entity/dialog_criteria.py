from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DialogCriteria(Base):
    """SQLAlchemy ORM model for dialog analysis criteria"""
    __tablename__ = 'dialog_criterias'

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    dialog_criteria_id = Column(String(50), nullable=False)
    dialog_row_fk_id = Column(PG_UUID(as_uuid=True), nullable=False)

    has_greeting = Column(Boolean, default=False)
    greeting_phrase = Column(String(255))

    has_name = Column(Boolean, default=False)
    found_name = Column(String(100))

    has_farewell = Column(Boolean, default=False)
    farewell_phrase = Column(String(255))

    interjections = Column(JSON)
    parasite_words = Column(JSON)
    abbreviations = Column(JSON)
    slang = Column(JSON)
    inappropriate_phrases = Column(JSON)

    has_diminutives = Column(Boolean, default=False)
    diminutives = Column(JSON)

    has_stopwords = Column(Boolean, default=False)
    stop_words = Column(JSON)

    has_swear_words = Column(Boolean, default=False)
    swear_words = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DialogCriteria(id={self.id}, dialog_row_fk_id={self.dialog_row_fk_id})>"