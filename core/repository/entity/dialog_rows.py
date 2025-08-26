from decimal import Decimal
from uuid import uuid4
from sqlalchemy import Column, String, Float
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DialogRow(Base):
    """SQLAlchemy ORM model for dialog rows"""
    __tablename__ = 'dialog_rows'

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    audio_dialog_fk_id = Column(PG_UUID(as_uuid=True), nullable=False)
    row_num = Column(String(20), nullable=False)
    row_text = Column(String, nullable=False)
    speaker_id = Column(String(50), nullable=False)
    start = Column(String(20), nullable=False)
    end = Column(String(20), nullable=False)
    mean_loudness = Column(Float, nullable=False)

    def __repr__(self):
        return f"<DialogRow(id={self.id}, row_num={self.row_num}, speaker={self.speaker_id})>"

    @classmethod
    def from_dict(cls, data: dict) -> 'DialogRow':
        """Create DialogRow instance from dictionary"""
        return cls(
            id=data.get('id', uuid4()),
            audio_dialog_fk_id=data['audio_dialog_fk_id'],
            row_num=data['row_num'],
            row_text=data['row_text'],
            speaker_id=data['speaker_id'],
            start=data['start'],
            end=data['end'],
            mean_loudness=data['mean_loudness'],
            has_swear_word=data.get('has_swear_word', False),
            has_greeting=data.get('has_greeting', False)
        )

    def print(self):
        print(f'{self.start}:{self.end}: [{self.speaker_id}] {self.row_text}')