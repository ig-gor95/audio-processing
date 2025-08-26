from typing import List, Dict
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

from core.config.datasource_config import DatabaseManager
from core.repository.entity.dialog_criteria import DialogCriteria
from core.repository.entity.dialog_rows import DialogRow

logger = logging.getLogger(__name__)


class DialogRowRepository:
    def __init__(self, engine=None):
        self.engine = engine or DatabaseManager.get_engine()
        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False
        )

    @contextmanager
    def _get_session(self):
        """Provide a transactional scope around operations"""
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

    def find_all(self) -> List[DialogRow]:
        """Get all dialog rows"""
        with self._get_session() as session:
            return session.query(DialogRow).all()

    def find_rows_without_criteria(self) -> List[DialogRow]:
        """Get all DialogRows that do NOT have any DialogCriteria"""
        with self._get_session() as session:
            return (
                session.query(DialogRow)
                .outerjoin(DialogCriteria, DialogRow.id == DialogCriteria.dialog_row_fk_id)
                .filter(DialogCriteria.dialog_criteria_id.is_(None))
                .filter(DialogRow.speaker_id != 'CLIENT')
                .all()
            )

    def save(self, row: DialogRow) -> DialogRow:
        with self._get_session() as session:
            if not isinstance(row, DialogRow):
                raise TypeError("Input must be a DialogRow instance")

            session.add(row)
            session.refresh(row)
            return row


    def save_bulk(self, dialog_rows: List[DialogRow]) -> None:
        """Save multiple DialogRow entities efficiently"""
        with self._get_session() as session:
            if not all(isinstance(row, DialogRow) for row in dialog_rows):
                raise TypeError("All items must be DialogRow instances")

            session.bulk_save_objects(dialog_rows)


    def delete_all_by_dialog_id(self, dialog_id: UUID) -> int:
        """Delete all rows for a specific audio dialog"""
        with self._get_session() as session:
            result = session.query(DialogRow) \
                .filter(DialogRow.audio_dialog_fk_id == dialog_id) \
                .delete()
            return result  # Returns number of rows deleted


    def find_by_dialog_id(self, dialog_id: UUID) -> List[DialogRow]:
        """Find all rows for a specific audio dialog"""
        with self._get_session() as session:
            return session.query(DialogRow) \
                .filter(DialogRow.audio_dialog_fk_id == dialog_id) \
                .all()


    def find_by_id(self, id: UUID) -> DialogRow:
        """Find all rows for a specific audio dialog"""
        with self._get_session() as session:
            return session.query(DialogRow) \
                .filter(DialogRow.id == id) \
                .first()


    def find_all(self) -> List[DialogRow]:
        with self._get_session() as session:
            return session.query(DialogRow).order_by(DialogRow.row_num).all()

    def update_speaker_id_by_id(self, dialog_id: int, new_speaker_id: str):
        with self._get_session() as session:
            dialog_row = session.query(DialogRow).filter(DialogRow.id == dialog_id).first()

            if dialog_row:
                if dialog_row.speaker_id == new_speaker_id:
                    return
                else:
                    print("updating")
                dialog_row.speaker_id = new_speaker_id
                session.commit()

    def update_loudness(self, row_id: UUID, loudness):
        with self._get_session() as session:
            update_query = text("""
                UPDATE dialog_rows 
                SET mean_loudness = :loudness 
                WHERE id = :row_id
            """)

            session.execute(
                update_query,
                {'loudness': loudness, 'row_id': str(row_id)}
            )
            session.commit()
