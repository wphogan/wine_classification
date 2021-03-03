# Wine Classifier

## Instructions:
- Install requirements: `pip install -r requirements.txt`
- Download [BERT base model (uncased)](https://huggingface.co/bert-base-uncased/tree/main) and place it in `/bert`
- To run train.py: `python train.py`
- To run train.py in debug mode: `python train.py --debug True`

#### Run best model:
- To run the best saved model, [download the checkpoint](https://drive.google.com/file/d/1CO0U1Kw5_kIimGJB9ehDMsd5Z6smlq-m/view?usp=sharing), unzip it, and place it in `/saved_models`.
- Then, run `python eval.py`.

#### Misc: 
- Logs for any given run are kept in `/log`
- The preprocessing script in `/preprocessing` combines the raw wine data and splits it randomly into train/dev/test splits (80/10/10). 