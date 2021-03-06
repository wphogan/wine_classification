'''
combine_and_split_data.py
Combine winemag-data-130k-v2 and winemag-data-150k file
Create 80/10/10 train/dev/test splits
Note: winemag-data-150k file has no data  for taster name, taster twitter handle, or wine title
'''

import pandas as pd

# Load CSV data file:
print('Preprocess start. Regular review text.')
df_130k = pd.read_csv('data/raw/winemag-data-130k-v2.csv', encoding='utf-8')
df_150k = pd.read_csv('data/raw/winemag-data_first150k.csv', encoding='utf-8')
frames = [df_130k, df_150k]
df_combined = pd.concat(frames, ignore_index=True)

# Split: 80/10/10 for train/dev/test
print('Combined set: {}'.format(len(df_combined)))
train = df_combined.sample(frac=0.80)  # 80%
dev_and_test = df_combined.drop(train.index)
dev = dev_and_test.sample(frac=0.50)  # 10%
test = dev_and_test.drop(dev.index)  # 10%
print('train/dev/test set sizes: {}, {}, {}'.format(len(train), len(dev), len(test)))

# Save splits
train.to_csv('data/train_v130k_and_150k.csv', index=False)
dev.to_csv('data/dev_v130k_and_150k.csv', index=False)
test.to_csv('data/test_v130k_and_150k.csv', index=False)
print('Preprocessing complete.')
