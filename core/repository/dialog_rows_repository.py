from typing import List, Dict
from contextlib import contextmanager
from uuid import UUID
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

from core.config.datasource_config import DatabaseManager
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

    def save(self, row_data: Dict) -> DialogRow:
        """Save a dialog row"""
        with self._get_session() as session:
            row = DialogRow(**row_data)
            session.add(row)
            return row

    def save_bulk(self, rows_data: List[Dict]) -> None:
        """Save multiple dialog rows efficiently"""
        with self._get_session() as session:
            rows = [DialogRow(**data) for data in rows_data]
            session.bulk_save_objects(rows)

    def delete_all_by_dialog_id(self, dialog_id: UUID) -> int:
        """Delete all rows for a specific audio dialog"""
        with self._get_session() as session:
            result = session.query(DialogRow)\
                .filter(DialogRow.audio_dialog_fk_id == dialog_id)\
                .delete()
            return result  # Returns number of rows deleted

    def find_by_dialog_id(self, dialog_id: UUID) -> List[DialogRow]:
        """Find all rows for a specific audio dialog"""
        with self._get_session() as session:
            return session.query(DialogRow)\
                .filter(DialogRow.audio_dialog_fk_id == dialog_id)\
                .all()

    def find_all(self) -> List[DialogRow]:
        with self._get_session() as session:
            return session.query(DialogRow).order_by(DialogRow.row_num).all()