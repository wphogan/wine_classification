import random
from datetime import datetime
import argparse

from numpy import inf
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Local Imports
from utils.data_handler import *
from utils.utils import *

# Set random seed
seed_val = random.randint(1, 10000)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Experiment identification
timestamp = datetime.now().strftime('%m%d_%H%M%S')
model_name = 'bert'
log_file = 'log/{}_{}.log'.format(model_name, timestamp)
logger = init_logger(model_name, log_file)


def load_tokenizer_and_model(device, n_categories):
    tok = AutoTokenizer.from_pretrained('bert/')
    model = BertForSequenceClassification.from_pretrained(
        'bert/',
        num_labels=int(n_categories),  # binary classification = 2
        output_attentions=False,
        output_hidden_states=False
    )
    return tok, model.to(device)


def run_validation_set(model, dev_dataloader, device):
    model.eval()
    predictions, model_name, eval_losses, true_labels = [], [], [], []
    for batch in dev_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   labels=b_labels.long())

        eval_losses.append(loss.item())

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        # Store predictions and true labels
        predictions.append(torch.argmax(logits, dim=1).numpy()[0])
        true_labels.append(label_ids.numpy()[0])

    # Report the final accuracy for this validation run.
    avg_val_accuracy = flat_accuracy(predictions, true_labels)

    # Calculate the average loss over all of the batches.
    avg_val_loss = np.mean(eval_losses)
    return avg_val_loss, avg_val_accuracy


def run_test_set(model, test_dataloader, device):
    logger.info('Predicting labels for test sentences...')
    model.eval()
    predictions, true_labels = [], []
    for batch in test_dataloader:
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu()
        label_ids = b_labels.to('cpu')

        # Store predictions and true labels
        predictions.append(torch.argmax(logits, dim=1).numpy()[0])
        true_labels.append(label_ids.numpy()[0])

    logger.info('Test set finished.')
    return predictions, true_labels


def analysis(model, test_dataloader, device, test_name):
    logger.info('\n\n\nResults for {}:'.format(test_name))
    logger.info('Test name: {}'.format(test_name))
    # Test model
    predictions, true_labels = run_test_set(model, test_dataloader, device)

    # Analysis
    acc = flat_accuracy(predictions, true_labels)
    precision_recall_f1(predictions, true_labels, logger)
    logger.info('Accuracy: {}'.format(acc))


def main(args):
    # Hyperparameters
    epochs = 10
    batch_size = 16
    max_len = 256 # 256
    tokenizer_name = 'bert-base-uncased'
    early_stop_epochs = 3
    learning_rate = 1e-5
    adam_epsilon = 1e-8
    set_optimizer = 'adam'  # sdg or adam
    experiment_title = 'Baseline Bert Experiment - No price, points, or taster_name - No Stop Words'

    record_exp(epochs, batch_size, max_len, learning_rate, adam_epsilon, tokenizer_name, experiment_title, logger,
               set_optimizer)

    # Directories
    make_directory('saved_models/' + timestamp)
    train_file = 'data/train_v130k.csv'
    dev_file = 'data/dev_v130k.csv'
    test_file = 'data/test_v130k.csv'

    # Init vars
    device = set_device()  # GPU / CPU
    best_val_loss = inf
    early_stop_threshold = early_stop_epochs

    # Initial data load to get categories:
    logger.info('Initial data loading:')
    label_encoder, n_categories = fit_labels(train_file, dev_file, test_file)
    x_train, y_train = initial_load(train_file, label_encoder)
    x_dev, y_dev = initial_load(dev_file, label_encoder)
    x_test, y_test = initial_load(test_file, label_encoder)
    save_labels(label_encoder, timestamp)

    # Shortened run for debugging
    if args.debug:
        logger.critical('WARNING: SHORTENED DEBUG MODEL IS RUNNING!')
        epochs = 1
        max_len = 24
        x_train = x_train[:200]
        y_train = y_train[:200]
        x_dev, y_dev = x_dev[:50], y_dev[:50]
        x_test, y_test = x_test[:50], y_test[:50]

    # Load tokenizer and model
    logger.info('Loading model.')
    tokenizer, model = load_tokenizer_and_model(device, n_categories)

    # Load training/dev/test data
    logger.info('Loading datasets.')
    train_dataset = load_dataset(tokenizer, max_len, x_train, y_train)
    dev_dataset = load_dataset(tokenizer, max_len, x_dev, y_dev)
    test_dataset = load_dataset(tokenizer, max_len, x_test, y_test)

    # Create dataloaders
    logger.info('Prepping dataloaders.')
    train_dataloader = gen_dataloaders(train_dataset, batch_size)
    dev_dataloader = gen_dataloader_test(dev_dataset, batch_size)
    test_dataloader = gen_dataloader_test(test_dataset, batch_size)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps
                                                )

    # Training Loop
    for epoch in range(0, epochs):
        logger.info('Epoch {}'.format(epoch))
        model.train()
        epoch_losses = []
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 labels=b_labels.long())

            epoch_losses.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Todo: uncomment this
            optimizer.step()
            scheduler.step()
            # **** END OF SINGLE BATCH **** #

        # Ave train & val loss for epoch 
        avg_train_loss = np.mean(epoch_losses)
        avg_val_loss, avg_val_accuracy = run_validation_set(model, dev_dataloader, device)

        # Record progress
        logger.info(
            f'Epoch:{epoch} | Train Loss:{avg_train_loss} | Val Loss:{avg_val_loss} | Val Acc:{avg_val_accuracy}')
        logger.info(f"training.loss:  {avg_train_loss}")
        logger.info(f"validation.loss:  {avg_val_loss}")
        logger.info(f"validation.accuracy:  {avg_val_accuracy}")

        # Save best model, tokenizer, and hyperparameters 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_threshold = early_stop_epochs  # reset counter
            save_model(timestamp, model, tokenizer, epochs, batch_size, max_len, learning_rate, adam_epsilon)
            logger.info("Saved best model.")
        else:
            early_stop_threshold -= 1

        # Early stopping
        logger.info(f'Epoch:{epoch} | Early stop:{early_stop_threshold}')
        if early_stop_threshold == 0:
            logger.info(f'STOPPING EARLY. EPOCH: {epoch}.')
            break

        # **** END OF EPOCH LOOP **** #

    # Training complete
    logger.info('Training complete.')

    # Dev-set:
    analysis(model, dev_dataloader, device, "Final dev-set performance:")

    # Test-set:
    analysis(model, test_dataloader, device, "Test-set performance:")
    # **** END OF RUN EXPERIMENT BLOCK **** #



if __name__ == '__main__':
    print('Begin BERT.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', required=False, default=False)
    main(parser.parse_args())
    print('End BERT.')
