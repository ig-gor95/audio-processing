from typing import Optional, Dict, List
from contextlib import contextmanager
from uuid import UUID
from sqlalchemy.orm import Session, sessionmaker, object_session
from sqlalchemy.exc import SQLAlchemyError
import logging

from core.config.datasource_config import DatabaseManager
from core.repository.entity.dialog_criteria import DialogCriteria  # You'll need to create this model

logger = logging.getLogger(__name__)

class DialogCriteriaRepository:
    def __init__(self, engine=None):
        self.engine = engine or DatabaseManager.get_engine()
        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False
        )

    def save_bulk(self, dialog_rows: List[DialogCriteria]) -> None:
        with self._get_session() as session:
            if not all(isinstance(row, DialogCriteria) for row in dialog_rows):
                raise TypeError("All items must be DialogCriteria instances")

            session.bulk_save_objects(dialog_rows)


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

    def save(self, criteria: DialogCriteria) -> DialogCriteria:
        with self._get_session() as session:
            if not isinstance(criteria, DialogCriteria):
                raise TypeError("Input must be a DialogCriteria instance")

            if object_session(criteria) is None:
                session.add(criteria)
            else:
                criteria = session.merge(criteria)
            session.flush()
            session.refresh(criteria)
            return criteria

    def delete_by_dialog_row_fk_id(self, dialog_row_id: UUID) -> int:
        with self._get_session() as session:
            result = session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_row_fk_id == dialog_row_id) \
                .delete()
            session.commit()
            return result

    def delete_by_id(self, id: UUID) -> int:
        with self._get_session() as session:
            result = session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_criteria_id == id) \
                .delete()
            session.commit()
            return result

    def find_by_criteria_id(self, criteria_id: str) -> Optional[DialogCriteria]:
        """Find criteria by its criteria_id."""
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_criteria_id == criteria_id) \
                .first()

    def find_all_by_row_fk_id(self, row_id: UUID) -> list[DialogCriteria]:
        """Find criteria by its criteria_id."""
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_row_fk_id == row_id) \
                .all()

    def find_by_row_fk_id(self, row_id: UUID) -> DialogCriteria:
        """Find criteria by its criteria_id."""
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_row_fk_id == row_id) \
                .first()

    def update_criteria(self, criteria_id: str, update_data: Dict) -> Optional[DialogCriteria]:
        """Update existing criteria."""
        with self._get_session() as session:
            criteria = session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_criteria_id == criteria_id) \
                .first()

            if criteria:
                for key, value in update_data.items():
                    setattr(criteria, key, value)

            return criteria

    def exists(self, dialog_row_id: UUID) -> bool:
        """Check if criteria exists for a dialog row."""
        with self._get_session() as session:
            return session.query(
                session.query(DialogCriteria)
                .filter(DialogCriteria.dialog_row_fk_id == dialog_row_id)
                .exists()
            ).scalar()