"""
*
* Acronym Resolution, Phase 3
* This combines phases 1 and 2 to create the end-to-end Acronym Resolver
*
"""
import random
import argparse
from transformers import AutoTokenizer, BertForSequenceClassification

# Local imports
from utils.data_handler import *
from utils.utils import *


def main(args):
    """
    Variables & Directories
    """
    # Saved model and results location
    ts = 'temp'
    t_name = 'test_v130k'
    test_file = 'data/{}.csv'.format(t_name)

    # Directories
    model_dir = 'saved_models/' + ts
    results_file = 'log/{}_results.txt'.format(ts)
    label_encoder_file = model_dir + '/label_encoder.pkl'
    print("Model: {}, Data: {}, Results file: {}".format(model_dir, test_file, results_file))
    open(results_file, 'w').close()  # clear file

    # Set random seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    """
    Load Model & Data
    """
    # Hyper-parameters
    batch_size = 1
    max_len = 128
    device = set_device()
    print('Using: {}'.format(device))

    # Load labels
    label_encoder = load_labels(label_encoder_file)
    n_labels = len(label_encoder.classes_)

    # Load model
    model_state_dict = torch.load(model_dir + '/pytorch_model.bin', map_location=torch.device(device))
    model = BertForSequenceClassification.from_pretrained(model_dir, state_dict=model_state_dict, num_labels=n_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load data
    x_test, y_test = initial_load(test_file, label_encoder)
    if args.debug:
        x_test, y_test = x_test[: 100], y_test[:100]
        print("DEBUG WARNING: SHORTENED TEST SET!!!!")
    test_dataset = load_dataset(tokenizer, max_len, x_test, y_test)
    test_dataloader = gen_dataloader_test(test_dataset, batch_size)
    predictions, true_labels = [], []

    """
    Predict
    """
    for batch in test_dataloader:
        input_ids, labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(input_ids, token_type_ids=None)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = labels.to('cpu')

        # Store predictions and true labels
        predictions.append(torch.argmax(logits, dim=1).numpy()[0])
        true_labels.append(label_ids.numpy()[0])

    """
    Analyze results
    """
    print('+' * 30, "RESULTS", '+' * 30)
    acc = flat_accuracy(predictions, true_labels)
    precision_recall_f1(predictions, true_labels)
    print('Accuracy: {}'.format(acc))


if __name__ == '__main__':
    print('Begin load and analyze.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    main(parser.parse_args())
    print('End load and analyze.')
