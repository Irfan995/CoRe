import os
import logging
import tensorflow as tf
from src.model_loader import ModelLoader
from src.training import Trainer
from data.dataset_loader import Dataset
from utils.helpers.methods import get_device, get_datetime_str
from utils.constants.location_constant import DATASET_PATH
from sentence_transformers import SentenceTransformer
from utils.constants.model_constant import BERT_MODEL_NAMES, SBERT_MODEL_NAMES

def main(dataset_key, model_type, pre_trained_model):
    # Load dataset
    dataset = Dataset(DATASET_PATH[dataset_key])
    
    # Load model and tokenizer
    loader = ModelLoader(model_type, pre_trained_model, dataset)
    model = loader.load_model()
    tokenizer = model.get_tokenizer()

    if model_type == 'saf' or model_type == 'slaf':
        vocab = model.get_vocab()

    training_eval_location = 'training_evaluation'
    if not os.path.isdir(training_eval_location):
        os.mkdir(training_eval_location)
    
    model_eval_location = training_eval_location + '/' + model_type
    if not os.path.isdir(model_eval_location):
        os.mkdir(model_eval_location)

    dataset_location = model_eval_location + '/' + dataset_key
    if not os.path.isdir(dataset_location):
        os.mkdir(dataset_location)
    
    pre_trained_model_location = dataset_location + '/' + pre_trained_model
    if not os.path.isdir(pre_trained_model_location):
        os.mkdir(pre_trained_model_location)

    saved_model_location = 'trained_model'
    if not os.path.isdir(saved_model_location):
        os.mkdir(saved_model_location)
    
    model_type_location = saved_model_location + '/' + model_type
    if not os.path.isdir(model_type_location):
        os.mkdir(model_type_location)
    
    pre_trained_model_location_trained_model = model_type_location + '/' + dataset_key
    if not os.path.isdir(pre_trained_model_location_trained_model):
        os.mkdir(pre_trained_model_location_trained_model)

    dataset_location_trained_model = pre_trained_model_location_trained_model + '/' + pre_trained_model
    if not os.path.isdir(dataset_location_trained_model):
        os.mkdir(dataset_location_trained_model)

    if model_type == 'sbert':
        logging.basicConfig(filename=pre_trained_model_location + '/' + SBERT_MODEL_NAMES[pre_trained_model]['name'] + '_' + get_datetime_str()  + '.log', level=logging.INFO)
    elif model_type == 'saf':
        logging.basicConfig(filename=pre_trained_model_location + '/' + SBERT_MODEL_NAMES[pre_trained_model]['name'] + '_' + get_datetime_str()  + '.log', level=logging.INFO)

    # Train and evaluate model
    if model_type == 'bert':
        # Split dataset into train and test
        train_dataset, test_dataset = dataset.split_data(tokenizer)
        trainer = Trainer(model.get_model(), tokenizer, train_dataset, test_dataset, pre_trained_model_location, dataset_location_trained_model)
        trainer.train()
        trainer.evaluate()
    elif model_type == 'sbert':
        # Split dataset into train and test
        train_x, test_x, train_y, test_y, total_classes, labels = dataset.split_data(sbert=tokenizer, test_size=0.2)
        trainer = Trainer(model.get_model(), eval_location=pre_trained_model_location, save_model_location=dataset_location_trained_model)
        trainer.train_sbert(total_classes, train_x, train_y, test_x, test_y, SBERT_MODEL_NAMES[pre_trained_model]['dimension'])
    elif model_type == 'saf':
        # Split dataset into train and test
        train_x, test_x, train_y, test_y, total_classes, labels = dataset.split_data(sbert=tokenizer, test_size=0.2)
        trainer = Trainer(model.get_model(), eval_location=pre_trained_model_location, save_model_location=dataset_location_trained_model)
        model = trainer.train_saf(total_classes, train_x, train_y, test_x, test_y, SBERT_MODEL_NAMES[pre_trained_model]['dimension'], vocab=vocab)
        
if __name__ == "__main__":
    dataset_key = 'CDS' # CDS -> Classroom Dataset ShortCommands
    model_type ='sbert'  # Change it to "bert"/ "saf" for training with bert or saf

    # for key in BERT_MODEL_NAMES:
    #     main(dataset_key, model_type, BERT_MODEL_NAMES[key])

    for key in SBERT_MODEL_NAMES:
        main(dataset_key, model_type, SBERT_MODEL_NAMES[key]['name'])