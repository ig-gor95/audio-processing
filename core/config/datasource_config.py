import json

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from typing import Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    _engine: Optional[Engine] = None

    @classmethod
    def get_engine(cls) -> Engine:
        if cls._engine is None:
            cls._initialize_engine()
        return cls._engine

    @classmethod
    def _initialize_engine(cls):
        try:
            from yaml_reader import ConfigLoader
            config = ConfigLoader("../configs/config.yaml")

            db_url = URL.create(
                drivername="postgresql+psycopg2",
                username=config.get('db.user'),
                password=config.get('db.password'),
                host=config.get('db.host'),
                port=config.get('db.port'),
                database=config.get('db.dbname')
            )

            cls._engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=5,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                pool_use_lifo=True,

                echo=False,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'your_app_name',
                    'client_encoding': 'utf8'
                },
                json_serializer=lambda x: json.dumps(x, ensure_ascii=False)
            )
            logger.info("Database engine initialized successfully")

        except Exception as e:
            logger.critical(f"Failed to initialize database engine: {e}")
            raise RuntimeError("Database connection failed") from e

    @classmethod
    def dispose_engine(cls):
        if cls._engine is not None:
            logger.info("Disposing database engine")
            cls._engine.dispose()
            cls._engine = None

    @classmethod
    @contextmanager
    def get_connection(cls):

        engine = cls.get_engine()
        conn = None
        try:
            conn = engine.connect()
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    @classmethod
    def get_pool_status(cls) -> dict:
        if cls._engine is None:
            return {}

        pool = cls._engine.pool
        return {
            'checked_out': pool.checkedout(),
            'checked_in': pool.checkedin(),
            'size': pool.size(),
            'overflow': pool.overflow()
        }