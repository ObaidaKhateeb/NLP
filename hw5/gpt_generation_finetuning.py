from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
import os
import sys

#choosing subset of a given dataset, if the subset already exists to load it (section 1)
def data_load(dataset_path):
    try: 
        subset = load_from_disk(dataset_path)
    except Exception as e:
        print(f'Failed to load the dataset from disk: {e}')
        exit(1)
    positive_review = subset.filter(lambda example: example["label"] == 1).shuffle(seed=42)
    negative_review = subset.filter(lambda example: example["label"] == 0).shuffle(seed=42)
    # to avoid the case of error as a result of less than 100 samples in one of the subsets. then it will choose as many samples as there are 
    positive_review_len = len(positive_review)
    negative_review_len = len(negative_review)
    positive_review = positive_review.select(range(min(100, positive_review_len)))
    negative_review = negative_review.select(range(min(100, negative_review_len)))
    return positive_review, negative_review

#A function that tokenizes the training dataset (section 3.2)
def tokenize_reviews(dataset, tokenizer, max_length=100):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset

#A function that generates reviews (section 3.9)
def reviews_generate(model, tokenizer, input_ids, attention_mask, max_length, temperature, top_p, repetition_penalty):
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, temperature=temperature, top_p=top_p, repetition_penalty = repetition_penalty, do_sample=True, num_return_sequences=5)
    generated_sentences = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_sentences = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(5)]
    return generated_sentences 

#A function that saves the generated reviews to a txt file (section 3.10)
def save_reviews(positive_reviews, negative_reviews, file_name):
    with open(file_name, 'w') as file:
        for type, reviews in [("positive", positive_reviews), ("negative", negative_reviews)]:
            file.write(f'Reviews generated by {type} model:\n')
            for i,review in enumerate(reviews):
                file.write(f'{i+1}. {review}\n')
            file.write(f'\n')


def main():
    dataset_path = sys.argv[1] #directory to the dataset
    generated_reviews_path = sys.argv[2] #directory to the reviewes txt file 
    save_directory = sys.argv[3]  #directory to save the model and tokenizer
    positive_review, negative_review = data_load(dataset_path) #loading the dataset

    #loading GPT-2 models and tokenizers (section 3.1)
    model_positive = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer_positive = GPT2Tokenizer.from_pretrained("gpt2")
    model_negative = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer_negative = GPT2Tokenizer.from_pretrained("gpt2")

    #tokenizing the datasets (section 3.2)
    tokenized_positive_dataset = tokenize_reviews(positive_review, tokenizer_positive)
    tokenized_negative_dataset = tokenize_reviews(negative_review, tokenizer_negative)

    #choosing the training arguments (section 3.3)
    training_positive_args = TrainingArguments(output_dir= '/tmp', do_eval = False, evaluation_strategy = "no", learning_rate= 2e-5, per_device_train_batch_size = 8, per_device_eval_batch_size = 8, num_train_epochs = 3, weight_decay = 0.01)
    training_negative_args = TrainingArguments(output_dir= '/tmp', do_eval = False, evaluation_strategy = "no", learning_rate= 2e-5, per_device_train_batch_size = 8, per_device_eval_batch_size = 8, num_train_epochs = 3, weight_decay = 0.01)

    #creating trainer (section 3.4)
    trainer_positive = Trainer(model=model_positive, args=training_positive_args, train_dataset=tokenized_positive_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_positive, mlm=False),)
    trainer_negative = Trainer(model=model_negative, args=training_negative_args, train_dataset=tokenized_negative_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer_negative, mlm=False),)

    #training the model (section 3.5)
    trainer_positive.train()
    trainer_negative.train()

    #saving the model and the tokenizer for future use (section 3.6)
    trainer_positive.model.save_pretrained(save_directory) 
    tokenizer_positive.save_pretrained(save_directory)
    trainer_negative.model.save_pretrained(save_directory)
    tokenizer_negative.save_pretrained(save_directory)

    #loading the model and the tokenizer (section 3.7)
    model_positive = GPT2LMHeadModel.from_pretrained(save_directory)
    tokenizer_positive = GPT2Tokenizer.from_pretrained(save_directory)
    model_negative = GPT2LMHeadModel.from_pretrained(save_directory)
    tokenizer_negative = GPT2Tokenizer.from_pretrained(save_directory)

    #creating input_ids and attention mask (section 3.8)
    prompt = "The movie was"
    input_ids_positive = tokenizer_positive.encode(prompt, return_tensors="pt")
    attention_mask_positive = input_ids_positive.ne(tokenizer_positive.pad_token_id)
    input_ids_negative = tokenizer_negative.encode(prompt, return_tensors="pt")
    attention_mask_negative = input_ids_negative.ne(tokenizer_negative.pad_token_id)

    #generating reviews (section 3.9) 
    positive_reviews = reviews_generate(model_positive, tokenizer_positive, input_ids_positive, attention_mask_positive, max_length=100, temperature=0.5, top_p=0.9, repetition_penalty=1.2)
    negative_reviews = reviews_generate(model_negative, tokenizer_negative, input_ids_negative, attention_mask_negative, max_length=100, temperature=0.5, top_p=0.9, repetition_penalty=1.2)

    #print the generated reviews to a txt file (section 3.10)  
    save_reviews(positive_reviews, negative_reviews, generated_reviews_path)

if __name__ == "__main__":
    main()
