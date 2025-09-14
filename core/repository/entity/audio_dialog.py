from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, Float, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AudioDialogStatus(Enum):
    NOT_PROCESSED = 'NOT_PROCESSED'
    PROCESSED = 'PROCESSED'
    ERROR = 'ERROR'

class AudioDialog(Base):
    """SQLAlchemy ORM model for audio dialogs"""
    __tablename__ = 'audio_dialogs'

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_name = Column(String(255), nullable=False)
    duration = Column(String(20), nullable=False)
    processing_time = Column(Float)
    status = Column(SQLEnum(AudioDialogStatus), default=AudioDialogStatus.NOT_PROCESSED)
    theme = Column(String(300))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AudioDialog(id={self.id}, filename='{self.file_name}', status={self.status})>"

    @property
    def filename_value(self) -> str:
        """Get the actual string value of the filename"""
        return self.file_name

    @filename_value.setter
    def filename_value(self, value: str):
        """Set the filename value"""
        self.file_name = value