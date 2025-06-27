from typing import Optional, Dict, List
from contextlib import contextmanager
from uuid import UUID
from sqlalchemy.orm import Session, sessionmaker
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

    def save(self, criteria_data: Dict) -> DialogCriteria:
        """Save dialog criteria to database using ORM."""
        with self._get_session() as session:
            criteria = DialogCriteria(**criteria_data)
            session.add(criteria)
            session.flush()
            return criteria

    def find_by_dialog_row(self, dialog_row_id: UUID) -> List[DialogCriteria]:
        """Find all criteria for a specific dialog row."""
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_row_fk_id == dialog_row_id) \
                .all()

    def find_by_criteria_id(self, criteria_id: str) -> Optional[DialogCriteria]:
        """Find criteria by its criteria_id."""
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_criteria_id == criteria_id) \
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