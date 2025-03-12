from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, concatenate_datasets
import sys

def data_load(dataset_path):
    try: 
        subset = load_from_disk(dataset_path)
    except Exception as e:
        print(f'Failed to load the dataset from disk: {e}')
        exit(1)
    try:
        #splitting the dataset into positive and negatvie 
        positive_review = subset.filter(lambda example: example["label"] == 1).shuffle(seed=42)
        negative_review = subset.filter(lambda example: example["label"] == 0).shuffle(seed=42)
        # to avoid the case of error as a result of less than 25 samples in one of the subsets. then it will choose as many samples as there are 
        positive_review_len = len(positive_review)
        negative_review_len = len(negative_review)
        positive_review = positive_review.select(range(min(25, positive_review_len)))
        negative_review = negative_review.select(range(min(25, negative_review_len)))
        dataset = concatenate_datasets([positive_review, negative_review]) #merge the two subsets into one 
    except Exception as e:
        print(f'Failed to process the dataset: {e}')
        exit(1)
    return dataset

def reviews_classify_helper(review, prompt, tokenizer, model):
    input_text = prompt + review
    inputs = tokenizer(input_text, return_tensors= 'pt')
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

def reviews_classify(dataset, zero_shot_prompt, few_shot_prompt, ins_based_prompt, model, tokenizer):
    results = []
    for idx, review in enumerate(dataset):
        text = review['text'] #extracting the review text
        true_label = 'positive' if review['label'] == 1 else 'negative' #extracting the true label of the review
        zero_shot_result = reviews_classify_helper(text, zero_shot_prompt, tokenizer, model)
        few_shot_result = reviews_classify_helper(text, few_shot_prompt, tokenizer, model)
        instruction_result = reviews_classify_helper(text, ins_based_prompt, tokenizer, model)
        results.append({"idx": idx+1, "Review": text, "true Label": true_label, "zero-shot": zero_shot_result, "few-shot": few_shot_result, "instruction-based": instruction_result})
    return results

def save_results(classification_results, results_path):
    try: 
        with open(results_path, "w") as file:
            for result in classification_results:
                file.write(f"Review {result['idx']}: {result['Review']}\n")
                file.write(f"Review {result['idx']} true label: {result['true Label']}\n")
                file.write(f"Review {result['idx']} zero-shot: {result['zero-shot']}\n")
                file.write(f"Review {result['idx']} few-shot: {result['few-shot']}\n")
                file.write(f"Review {result['idx']} instruction-based: {result['instruction-based']}\n\n")
    except Exception as e:
        print(f'Failed to save the results to the disk: {e}')

def main():
    dataset_path = sys.argv[1]
    results_path = sys.argv[2] 

    #loading the model and its tokinizer (section 4.1)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small") 
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    #choosing 50 balanced reviews (section 4.2)
    dataset = data_load(dataset_path) #loading the dataset

    #creating the prompts (section 4.3)
    zero_shot_prompt = "Classify the following movie reviews as positive or negative. Review:"
    few_shot_prompt = "Classify the following movie reviews as 'positive' or 'negative':\n1. 'The movie was absolutely fantastic, I couldn't stop watching!' -> positive\n2. 'It was a complete waste of time, I regret watching it.' -> negative\n Review:"
    ins_based_prompt = "You are a movie reviews classifier: Please classify the following reviews as positive or negative. Review:" 

    #classifying the reviews and saving the results to the given file (section 4.4)
    classification_results = reviews_classify(dataset, zero_shot_prompt, few_shot_prompt, ins_based_prompt, model, tokenizer)
    save_results(classification_results, results_path)
    
    #calculating the accuracy of the model (section 4.5)
    #from sklearn.metrics import accuracy_score 
    #true_labels = [review['true Label'] for review in classification_results]
    #for prompt in ['zero-shot', 'few-shot', 'instruction-based']:
    #    predicted_labels = [review[prompt] for review in classification_results]
    #    accuracy = accuracy_score(true_labels, predicted_labels)
    #    print(f"Accuracy of {prompt} prompt: {accuracy}")







if __name__ == "__main__":
    main()