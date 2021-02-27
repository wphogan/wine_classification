import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset


def inspect(df):
    print("HEAD:\n", df.head())
    print("DESCRIBE:\n", df.describe())
    print("COLS:\n", df.columns)
    print("ALL:\n", df)


def initial_load(file_name):
    # Init label encoder:
    label_encoder = LabelEncoder()

    # Load the dataset into a dataframe
    df = pd.read_csv(file_name, encoding='utf-8')
    df['variety'] = pd.Categorical(df['variety']).codes
    label_encoder.fit(df['variety'])
    x = list(df['description'])
    y = list(label_encoder.transform(df['variety']))
    n_categories = max(y) + 1  # max categorical number plus 1 for zero category
    return x, y, n_categories, label_encoder


def load_dataset(tokenizer, max_len, x, y):
    input_ids_all, labels_all = tokenize(tokenizer, x, y, max_len)
    return TensorDataset(input_ids_all, labels_all)


def gen_dataloader_test(test_dataset, batch_size):
    # For test the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
        test_dataset,  # The validation samples.
        sampler=SequentialSampler(test_dataset),  # Pull predictions batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return test_dataloader


def gen_dataloaders(train_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return train_dataloader


def tokenize(tokenizer, documents, labels, max_len):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    labels_list = []

    # For every sentence...
    n_docs = len(documents)
    for i, doc in enumerate(documents):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            doc,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # Labels to include in test/training
        labels_list.append(labels[i])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels_list)
    return input_ids, labels
