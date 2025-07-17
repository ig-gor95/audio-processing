from typing import Optional, List
from contextlib import contextmanager
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

from core.config.datasource_config import DatabaseManager
from core.repository.entity.audio_dialog import AudioDialog, AudioDialogStatus

logger = logging.getLogger(__name__)

class AudioDialogRepository:
    def __init__(self, engine=None):
        self.engine = engine or DatabaseManager.get_engine()
        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False
        )

    @contextmanager
    def _get_session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            session.close()

    def find_by_filename(self, file_name: str, load_attributes: bool = True) -> Optional[AudioDialog]:
        with self._get_session() as session:
            result = session.query(AudioDialog) \
                .filter(AudioDialog.file_name == file_name) \
                .first()

            if result and load_attributes:
                session.refresh(result)

            return result

    def save(self, audio_dialog: AudioDialog) -> None:
        """Save an AudioDialog entity using ORM."""
        with self._get_session() as session:
            session.add(audio_dialog)

    def update_status(self, file_id: str, new_status: AudioDialogStatus, processed_time: float) -> None:
        """Update processing status and time using ORM."""
        with self._get_session() as session:
            dialog = session.query(AudioDialog).get(file_id)
            if dialog:
                dialog.status = new_status
                dialog.processing_time = processed_time

    def find_all(self) -> List[AudioDialog]:
        with self._get_session() as session:
            return session.query(AudioDialog).all()

    def exists(self, file_name: str) -> bool:
        """Check if a file exists using ORM."""
        with self._get_session() as session:
            return session.query(
                session.query(AudioDialog)
                .filter(AudioDialog.file_name == file_name)
                .exists()
            ).scalar()