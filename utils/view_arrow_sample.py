import pyarrow.ipc as ipc
from datasets import Dataset

file_path='/data/xgao/code/interpretability/SAELens-V/data/processed_dataset/batch_1/data-00000-of-00001.arrow'

ds = Dataset.from_file(file_path)
print(ds)
print('input_ids_size:', len(ds[0]['input_ids'][0]))
print('attention_mask_size:', len(ds[0]['attention_mask'][0]))
print('pixel_values:', len(ds[0]['pixel_values'][0][0][0][0]))
print('image_sizes:', ds[0]['image_sizes'][0])