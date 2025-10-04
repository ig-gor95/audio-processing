from typing import Optional, List
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

from core.config.datasource_config import DatabaseManager
from core.repository.entity.audio_dialog import AudioDialog, AudioDialogStatus
import pandas as pd

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

    def find_by_id(self, id: UUID, load_attributes: bool = True) -> Optional[AudioDialog]:
        with self._get_session() as session:
            result = session.query(AudioDialog) \
                .filter(AudioDialog.id == id) \
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

    def update_theme(self, file_id: str, theme: str) -> None:
        """Update processing status and time using ORM."""
        with self._get_session() as session:
            dialog = session.query(AudioDialog).get(file_id)
            if dialog:
                dialog.theme = theme

    def update_llm_data(self, file_id: str, new_data: dict) -> None:
        """Update LLM metadata in JSONB column."""
        with self._get_session() as session:
            dialog = session.query(AudioDialog).get(file_id)
            if dialog:
                current = dialog.llm_data_short or {}
                current.update(new_data)  # аккуратное слияние
                dialog.llm_data_short = current
                session.add(dialog)
                session.commit()

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

    def get_all_for_report(self):
        return pd.read_sql("""
            SELECT *
            from dialog_criterias t1
            join dialog_rows t2 on t1.dialog_row_fk_id::uuid = t2.id
            left join audio_dialogs t3 on t2.audio_dialog_fk_id = t3.id
            where t3.updated_at >= '2025-09-03'
            """, self.engine)