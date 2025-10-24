import uuid
from typing import Optional, Dict, List
from contextlib import contextmanager
from uuid import UUID

import pandas as pd
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

    def save_pd(self, df: pd.DataFrame):
        df.to_sql(
            'dialog_criterias',
            self.engine,
            if_exists='append',
            index=False
        )

    def pd_get_all_unprocessed_rows(self, dialog_ids: Optional[List[uuid.UUID]] = None) -> pd.DataFrame:
        """
        Get all unprocessed dialog rows, optionally filtered by dialog IDs.
        
        Args:
            dialog_ids: Optional list of audio_dialog UUIDs to filter by
            
        Returns:
            DataFrame with unprocessed dialog rows
        """
        query = """
            SELECT row.*
            FROM dialog_rows row
            LEFT JOIN dialog_criterias c ON c.dialog_row_fk_id = row.id
            WHERE row.row_text IS NOT NULL 
              AND row.row_text != ' ' 
              AND row.row_text != ''
              AND c.dialog_criteria_id IS NULL
        """
        
        if dialog_ids:
            logger.debug(f"Filtering by {len(dialog_ids)} specific dialog IDs")
            uuid_strings = ','.join([f"'{str(uid)}'" for uid in dialog_ids])
            query += f" AND row.audio_dialog_fk_id IN ({uuid_strings})"
        
        return pd.read_sql(query, self.engine)

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

    def find_by_row_fk_id_in(self, row_ids: List[UUID]) -> List[DialogCriteria]:
        with self._get_session() as session:
            return session.query(DialogCriteria) \
                .filter(DialogCriteria.dialog_row_fk_id.in_(row_ids)) \
                .all()

    def update_all_criteria(self, criteria_list: List[DialogCriteria]) -> List[DialogCriteria]:
        if not criteria_list:
            return []

        with self._get_session() as session:
            results = []

            for criteria in criteria_list:
                if getattr(criteria, 'id', None):
                    persisted = session.merge(criteria)
                else:
                    persisted = criteria
                    session.add(persisted)

                results.append(persisted)

            try:
                session.flush()
                for entity in results:
                    session.refresh(entity)
                return results
            except SQLAlchemyError as e:
                session.rollback()
                raise RuntimeError(f"Batch update failed: {str(e)}") from e

    def exists(self, dialog_row_id: UUID) -> bool:
        """Check if criteria exists for a dialog row."""
        with self._get_session() as session:
            return session.query(
                session.query(DialogCriteria)
                .filter(DialogCriteria.dialog_row_fk_id == dialog_row_id)
                .exists()
            ).scalar()
