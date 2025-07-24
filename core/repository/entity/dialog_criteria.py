from uuid import UUID, uuid4
from sqlalchemy import Column, String, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DialogCriteria(Base):
    """SQLAlchemy ORM model for dialog analysis criteria"""
    __tablename__ = 'dialog_criterias'

    dialog_criteria_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    dialog_row_fk_id = Column(PG_UUID(as_uuid=True), nullable=False)

    greeting_phrase = Column(String(255))

    found_name = Column(String(100))

    farewell_phrase = Column(String(255))

    interjections = Column(JSON)
    parasite_words = Column(JSON)
    abbreviations = Column(JSON)
    slang = Column(JSON)
    inappropriate_phrases = Column(JSON)
    non_professional_phrases = Column(JSON)

    diminutives = Column(JSON)
    await_requests = Column(JSON)

    stop_words = Column(JSON)

    swear_words = Column(JSON)

    order_offer = Column(JSON)
    order_processing = Column(JSON)
    order_resume = Column(JSON)

    def __repr__(self):
        return f"<DialogCriteria(id={self.dialog_criteria_id}, dialog_row_fk_id={self.dialog_row_fk_id})>"