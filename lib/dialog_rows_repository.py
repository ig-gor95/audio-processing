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


DELETE_ALL_BY_DIALOG_ID_STMT = text("""
        DELETE FROM dialog_rows 
        WHERE audio_dialog_fk_id = :audio_dialog_fk_id
    """)


def do_select_one_request(stmt, params: Dict) -> dict:
    with engine.connect() as conn:
        result = conn.execute(stmt, params).fetchone()
        return dict(result._asdict()) if result else None

def save(row_id,
         audio_dialog_fk_id,
         row_num,
         row_text,
         speaker_id,
         start,
         end,
         has_swear_word=False,
         has_greeting=False):
    data = [{
        'id': row_id,
        'audio_dialog_fk_id': audio_dialog_fk_id,
        'row_num': row_num,
        'row_text': row_text,
        'speaker_id': speaker_id,
        'start': start,
        'end': end,
        'has_swear_word': has_swear_word,
        'has_greeting': has_greeting
    }]
    pd.DataFrame(data).to_sql(
        name='dialog_rows',
        con=engine,
        if_exists='append',
        index=False
    )


def delete_all_by_dialog_id(dialog_id):
    with engine.connect() as conn:
        conn.execute(
            statement=DELETE_ALL_BY_DIALOG_ID_STMT,
            parameters={'audio_dialog_fk_id': dialog_id}
        )
        conn.commit()
