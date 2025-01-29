import os 
import sys
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#choosing subset of a given dataset, if the subset already exists to load it (section 1)
def data_load(dataset_path):
    if os.path.exists(os.path.join(dataset_path, 'dataset_info.json')): #if there's already subset in the disk load it 
        try: 
            subset = load_from_disk(dataset_path)
        except Exception as e:
            print(f'Failed to load the dataset from disk: {e}')
            exit(1)
    else: 
        dataset = load_dataset('imdb') #loading the IMDB dataset
        subset = dataset["train"].shuffle(seed=42).select(range(500)) #choosing 500 samples randomly 
        try: 
            subset.save_to_disk(dataset_path) #saving the subset to the disk 
        except Exception as e:
            print(f'Failed to save the dataset to the disk: {e}')
            return subset 
        try:
            subset = load_from_disk(dataset_path)
        except Exception as e:
            print(f'Failed to load the dataset from disk: {e}')
            exit(1)
    return subset

#function used in computing the accuracy of the model
def compute_metrics(eval_pred):
    logits, labels = eval_pred 
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    #choosing subset of the IMDB dataset (section 1)
    dataset_path = sys.argv[1] 
    subset = data_load(dataset_path)
    
    #loading bert-base-uncased model (section 2.1)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)

    #tokenizing the text data (section 2.2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenization(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    subset = subset.map(tokenization, batched=True)
    subset = subset.rename_column("label", "labels")

    #splittting the dataset into training set and validation set (section 2.3)
    subset = subset.train_test_split(test_size=0.2, seed=42)
    train_set = subset["train"]
    val_set = subset["test"]

    #training the model (section 2.4)
    training_args = TrainingArguments(output_dir= './results', evaluation_strategy = "no", learning_rate= 2e-5, per_device_train_batch_size = 8, per_device_eval_batch_size = 8, num_train_epochs = 3, weight_decay = 0.01)
    trainer = Trainer(model = model, args = training_args, train_dataset = train_set, eval_dataset = val_set, tokenizer = tokenizer, compute_metrics=compute_metrics)
    trainer.train()

    #evaluating the model (section 2.5) 
    accuracy = trainer.evaluate()
    print(accuracy)

if __name__ == "__main__":
    main()

