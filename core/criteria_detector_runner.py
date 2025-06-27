import pandas as pd

from core.config.datasource_config import DatabaseManager
from core.post_processors.criteria_detector import process_rows_parallel
from core.repository import dialog_rows_repository
from yaml_reader import ConfigLoader

if __name__ == "__main__":
    rows = dialog_rows_repository.find_all()
    count = 0
    print(len(rows))
    config = ConfigLoader("../../configs/config.yaml")
    engine = DatabaseManager.get_engine()
    b = []
    data = process_rows_parallel(b)
    df = pd.DataFrame(data)
    df.to_sql(
        name='dialog_criterias',  # Table name
        con=engine,
        if_exists='append',  # Append to existing table
        index=False,  # Don't write row index
        method='multi',  # Multi-row insert
        chunksize=100  # Batch size
    )