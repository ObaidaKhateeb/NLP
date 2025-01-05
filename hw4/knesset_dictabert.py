from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

#A methid that extracts the sentences from txt file
def sentences_extract(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

#A method that predicts the masked tokens in a sentence
def masked_tokens_predict(sentence, tokenizer, model):
    masked_sentence = sentence.replace("[*]", '[MASK]') #replacing [*] by [MASK]
    inputs = tokenizer(masked_sentence, return_tensors="pt") #tokenizing the sentence
    
    # Predict the masked tokens
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    #extracting the predicted tokens
    mask_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    predicted_tokens = [tokenizer.decode(predictions[0, idx].topk(1).indices).strip() for idx in mask_indices]

    return predicted_tokens

#A method that replaces the masked tokens with the predicted ones
def masked_replace(masked_sentences, predicted_tokens):
    masked_sentence_splitted = masked_sentences.split()
    j = 0
    for i,token in enumerate(masked_sentence_splitted):
        if token == '[*]' and j < len(predicted_tokens):
            masked_sentence_splitted[i] = predicted_tokens[j]
            j += 1
    return ' '.join(masked_sentence_splitted)

def main():
    original_sentences_file = "original_sampled_sents.txt"
    masked_sentences_file = "masked_sampled_sents.txt"
    output_path = ""

    output_file = os.path.join(output_path, 'dictabert_results.txt')
    
    #Extracting the sentence out of the txt files 
    original_sentences = sentences_extract(original_sentences_file)
    masked_sentences = sentences_extract(masked_sentences_file)

    #Loading the DictaBERT model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')

    #Predicting the masked tokens and writing the results to the txt file 
    with open(output_file, 'w', encoding='utf-8') as file:
        for original_sentence, masked_sentence in zip(original_sentences, masked_sentences):
            file.write(f"original_sentence: {original_sentence}\n")
            file.write(f"masked_sentence: {masked_sentence}\n")
            predicted_tokens = masked_tokens_predict(masked_sentence, tokenizer, model)
            new_sentence = masked_replace(masked_sentence, predicted_tokens)
            file.write(f"dictaBERT_sentence: {new_sentence}\n")
            file.write(f"dictaBERT tokens: {','.join(predicted_tokens)}\n")            

if __name__ == "__main__":
    main()
