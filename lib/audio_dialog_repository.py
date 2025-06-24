from datetime import datetime
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine, text

from yaml_reader import ConfigLoader

config = ConfigLoader("../configs/config.yaml")

engine = create_engine(
    f"postgresql+psycopg2://{config.get("db.user")}:{config.get("db.password")}@"
    f"{config.get("db.host")}:{config.get("db.port")}/{config.get("db.dbname")}"
)

FIND_BY_NAME_STMT = text("""
        SELECT id, file_name, status, duration, created_at 
        FROM audio_dialog 
        WHERE file_name = :file_name
    """)
UPDATE_STATUS_AND_PROCESSING_TIME = text("""
        UPDATE audio_dialog 
        SET status = :status, 
            processing_time = :processed_time
        WHERE id = :file_id
    """)


def find_by_filename(file_name: str) -> dict:
    return do_select_one_request(FIND_BY_NAME_STMT, {'file_name': file_name})


def do_select_one_request(stmt, params: Dict) -> dict:
    with engine.connect() as conn:
        result = conn.execute(stmt, params).fetchone()
        return dict(result._asdict()) if result else None


def save(dialog_id, file_name, duration, processing_time=None, status='NOT_PROCESSED'):
    data = [{
        'id': dialog_id,
        'file_name': file_name,
        'status': status,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration,
        'processing_time': processing_time,
    }]
    pd.DataFrame(data).to_sql(
        name='audio_dialog',
        con=engine,
        if_exists='append',
        index=False
    )

def update_status(file_id, new_status, processed_time):
    with engine.connect() as conn:
        conn.execute(
            statement=UPDATE_STATUS_AND_PROCESSING_TIME,
            parameters={'file_id': file_id, 'status': new_status, 'processed_time': processed_time}
        )
        conn.commit()
