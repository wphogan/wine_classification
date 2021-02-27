import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score


def csv_to_list_x(filename):
    '''Text to list'''
    df = pd.read_csv(filename)
    x = list(df['text'])
    return x


def init_logger(model_name, log_filename):
    logger = logging.getLogger(model_name)
    hdlr = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def preds_to_csv_file(preds, time_stamp, model_name='bert', debug=False):
    if debug:
        output_file = "predictions/debug_{}_{}_predicted.csv".format(time_stamp, model_name)
    else:
        output_file = "predictions/{}_{}_predicted.csv".format(time_stamp, model_name)
    dic = {"Id": [], "Predicted": []}
    for i, pred in enumerate(preds):
        dic["Id"].append(i)
        dic["Predicted"].append(pred)

    dic_df = pd.DataFrame.from_dict(dic)
    dic_df.to_csv(output_file, index=False)
    print("SAVED PREDICTION FILE: ", output_file)


def csv_to_list_y(filename):
    '''Labels to list'''
    df = pd.read_csv(filename)
    y = list(pd.Categorical(df['label']).codes)
    return y


def set_device():
    if torch.cuda.is_available():
        print("Nice! Using GPU.")
        return 'cuda'
    else:
        print("Watch predictions! Using CPU.")
        return 'cpu'


def make_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception:
        print("Error -- could not create directory.")


def save_labels(le, ts):
    output = open('saved_models/{}/label_encoder.pkl'.format(ts), 'wb')
    pickle.dump(le, output)
    output.close()


def load_labels(label_file):
    pkl_file = open(label_file, 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    return le


def save_model(time_stamp, model, tokenizer, epochs, batch_size, max_len, learning_rate, adam_epsilon):
    model_dir = 'saved_models/{}/'.format(time_stamp)
    args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'max_len': max_len,
        'learning_rate': learning_rate,
        'adam_epsilon': adam_epsilon
    }
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(model_dir, 'training_args.bin'))


def annot_confusion_matrix(valid_tags, pred_tags):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


def record_exp(epochs, batch_size, max_len, learning_rate, adam_epsilon, tokenizer_name, experiment_title, logger,
               set_optimizer=False, net=False):
    print(f'++++++++++++{experiment_title}++++++++++++')
    print(f'Epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'max_len: {max_len}')
    print(f'learning_rate: {learning_rate}')
    print(f'tokenizer name: {tokenizer_name}')
    print(f'adam_epsilon: {adam_epsilon}')
    print(f'optimizer: {set_optimizer}')

    logger.info(f'++++++++++++{experiment_title}++++++++++++')
    logger.info(f'Epochs: {epochs}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'max_len: {max_len}')
    logger.info(f'learning_rate: {learning_rate}')
    logger.info(f'tokenizer name: {tokenizer_name}')
    logger.info(f'adam_epsilon: {adam_epsilon}')
    logger.info(f'optimizer: {set_optimizer}')
    if net:
        logger.info(f'net: {net}')
        print(f'net: {net}')


def flatten(predictions, true_labels, main_run=False):
    if main_run:
        test_preds = np.concatenate(predictions, axis=0)
        test_labels = np.concatenate(true_labels, axis=0)
        test_preds = np.argmax(test_preds, axis=1).flatten()
        test_labels = test_labels.flatten()
    else:
        test_preds = np.argmax(np.asarray(predictions).squeeze(), axis=1)
        test_labels = np.asarray(true_labels).squeeze()
    return test_preds, test_labels


def flatten_logits(logits, doc_ids, main_run=False):
    if main_run:
        try:
            logits_flat = np.concatenate(logits, axis=0)
            ids_flat = np.concatenate(doc_ids, axis=0)
        except:
            print('Flatten exception')
            logits_flat = np.concatenate(np.asarray(logits), axis=0)
            temp = []
            for id in doc_ids:
                for single_id in id:
                    temp.append(single_id.cpu())
            print('temp len: ', len(temp))
            ids_flat = np.asarray(temp)

    else:
        print('not main run!!')
        logits_flat = np.concatenate(logits, axis=0)
        ids_flat = np.concatenate(doc_ids, axis=0)
        # ORIG:
        # logits_flat = np.asarray(logits).squeeze()
        # ids_flat = np.asarray(doc_ids).squeeze()
    return logits_flat, ids_flat


def index_to_text(test_preds, test_labels, le):
    pred_flat_text = list(le.inverse_transform(test_preds))
    label_flat_text = list(le.inverse_transform(test_labels))
    return pred_flat_text, label_flat_text


def confusion_classification(test_preds, test_labels, le, logger):
    pred_flat_text, label_flat_text = index_to_text(test_preds, test_labels, le)
    cl_report = classification_report(label_flat_text, pred_flat_text)
    conf_mat = annot_confusion_matrix(label_flat_text, pred_flat_text)
    logger.info(f"Classification Report:\n {cl_report}")
    logger.info(f"Confusion Matrix:\n {conf_mat}")


def flat_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)


def metrics(title, truths, preds):
    '''Calc accuracy, f1 macro, f1 micro'''
    accuracy = accuracy_score(truths, preds)
    f1_macro = f1_score(truths, preds, average='macro')
    f1_micro = f1_score(truths, preds, average='micro')
    print("\n{}\nAccuracy: {}, F1-Macro: {}, F1-Micro: {}".format(title, accuracy, f1_macro, f1_micro))


def precision_recall_f1(y_pred, y_true, logger=False):
    scores = {'p_micro': precision_score(y_true, y_pred, average='micro'),
              'r_micro': recall_score(y_true, y_pred, average='micro'),
              'f1_micro': f1_score(y_true, y_pred, average='micro'),
              'p_macro': precision_score(y_true, y_pred, average='macro'),
              'r_macro': recall_score(y_true, y_pred, average='macro'),
              'f1_macro': f1_score(y_true, y_pred, average='macro')}
    print(f"test.p_micro  {scores['p_micro']}")
    print(f"test.r_micro {scores['r_micro']}")
    print(f"test.f1_micro {scores['f1_micro']}")
    print(f"test.p_macro {scores['p_macro']}")
    print(f"test.r_macro {scores['r_macro']}")
    print(f"test.f1_macro {scores['f1_macro']}")

    if logger:
        logger.info(f"test.p_micro  {scores['p_micro']}")
        logger.info(f"test.r_micro {scores['r_micro']}")
        logger.info(f"test.f1_micro {scores['f1_micro']}")

        logger.info(f"test.p_macro {scores['p_macro']}")
        logger.info(f"test.r_macro {scores['r_macro']}")
        logger.info(f"test.f1_macro {scores['f1_macro']}")
