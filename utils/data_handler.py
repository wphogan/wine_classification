import pandas as pd
import torch
import re
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from nltk.corpus import stopwords


def fit_labels(f_train, f_dev, f_test):
    label_encoder = LabelEncoder()
    df_train = pd.read_csv(f_train, encoding='utf-8')
    df_dev = pd.read_csv(f_dev, encoding='utf-8')
    df_test = pd.read_csv(f_test, encoding='utf-8')
    frames = [df_train, df_dev, df_test]
    df_combined = pd.concat(frames, ignore_index=True)
    df_combined = df_combined.fillna('')  # replace nan with an empty string
    label_encoder.fit(df_combined['variety'])
    n_categories = len(label_encoder.classes_)
    return label_encoder, n_categories


def initial_load(file_name, label_encoder, no_stop_words=True):
    # Load the dataset into a dataframe
    df = pd.read_csv(file_name, encoding='utf-8')
    df = df.fillna('')  # replace nan with an empty string

    # CHOOSE THE MODELS FEATURES HERE:
    # All fields:
    # df['model_features'] = df['country'] + ' ' + df['designation'] + ' ' + df['province'] + ' ' + df['region_1'] + ' ' + df[
    #     'region_2'] + ' ' + str(df['price']) + ' ' + df['winery'] + ' ' + df['taster_name'] + ' ' + str(
    #     df['points']) + ' ' + df['title'] + ' ' + df['description'] # All fields

    # Without price, points, and taster_name
    df['model_features'] = df['country'] + ' ' + df['designation'] + ' ' + df['province'] + ' ' + df['region_1'] + ' ' + \
                           df[
                               'region_2'] + ' ' + df['winery'] + ' ' + df['title'] + ' ' + df['description']

    # Just the title
    # df['model_features'] = df['title']

    # Just the description
    # df['model_features'] = df['description']

    # Remove stop words
    if no_stop_words:
        x = remove_stop_words(df['model_features'])
    else:
        x = list(df['model_features'])

    y = list(label_encoder.transform(df['variety']))
    return x, y


def remove_stop_words(df):
    sw = stopwords.words('english')
    list_of_words = []
    for phase_word in df:
        list_of_words.append(
            ' '.join([re.sub('[^a-zA-Z0-9]', '', word) for word in phase_word.split() if not word in sw]))
    x = list_of_words
    return x


def load_dataset(tokenizer, max_len, x, y):
    input_ids_all, labels_all = tokenize(tokenizer, x, y, max_len)
    return TensorDataset(input_ids_all, labels_all)


def gen_dataloader_test(test_dataset, batch_size):
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
