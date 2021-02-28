'''
split_data.py
Create train/dev/test splits using winemag-data-130k-v2  only
This file has the most complete meta data compared to the winemag-data-150k file
'''

import pandas as pd

# Load CSV data file:
print('Preprocess start. Regular review text.')
df_130k = pd.read_csv('data/raw/winemag-data-130k-v2.csv', encoding='utf-8')

# Split: 80/10/10 for train/dev/test
print('Combined set: {}'.format(len(df_130k)))
train = df_130k.sample(frac=0.80)  # 80%
dev_and_test = df_130k.drop(train.index)
dev = dev_and_test.sample(frac=0.50)  # 10%
test = dev_and_test.drop(dev.index)  # 10%
print('train/dev/test set sizes: {}, {}, {}'.format(len(train), len(dev), len(test)))

# Save splits
train.to_csv('data/train_v130k.csv', index=False)
dev.to_csv('data/dev_v130k.csv', index=False)
test.to_csv('data/test_v130k.csv', index=False)
print('Preprocessing complete.')