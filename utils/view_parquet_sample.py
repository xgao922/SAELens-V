import pandas as pd
import pyarrow.parquet as pq
from pprint import pprint

file_path = '/data/xgao/code/interpretability/SAELens-V/data/test/test.parquet'
# file_path = '/data/xgao/code/interpretability/SAELens-V/data/processed_dataset/batch_1/data-00000-of-00001.arrow'
df = pd.read_parquet(file_path)
print(df.iloc[0])
print('\n')

table = pq.read_table(file_path)
print(table.to_pylist()[0])

