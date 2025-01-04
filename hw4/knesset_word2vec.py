from gensim.models import Word2Vec
import json
import sys

#A method that extracts the json lines from a JSONL file (section 1)
def json_lines_extract(file):
    try: 
        with open(file, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        print('The file {file} is not found')
        sys.exit(1)

#A helper method that checks if a token is a Hebrew word (section 1.1)
def is_word(token):
    hebrew_letters = {'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ך', 'ל', 'מ', 'ם','נ', 'ן', 'ס', 'ע', 'פ', 'ף', 'צ', 'ץ','ק', 'ר', 'ש', 'ת'}
    valid_start = token[0] in hebrew_letters
    valid_end = token[-1] in hebrew_letters or token[-1] == "'"
    valid_middle = all((char in hebrew_letters) or (char in ['"', "”","'"]) for char in token[1:-1])
    if valid_start and valid_end and valid_middle:
        return True
    else:
        if token[0] == '"' and token[-1] == '"' and len(token) > 2 and is_word(token[1:-1]):
            return True
        elif token[0] == "'" and token[-1] == "'" and len(token) > 2 and is_word(token[1:-1]):
            return True
        else:
            return False

#A helper method that update the tokenize sentence to have only words (section 1.1)
def keep_only_words(tokenized_sentence):
    updated_tokenized_sentence = []
    for token in tokenized_sentence:
        if is_word(token):
            if token[0] in ['"', '”', "'"] and token[-1] == token[0] and len(token) > 2:
                updated_tokenized_sentence.append(token[1:-1])
            else:
                updated_tokenized_sentence.append(token)
    return updated_tokenized_sentence

#A method that extract from each json line an only words tokenized sentence (section 1.1)
def json_lines_to_tokens(json_lines):
    tokenized_sentences = []
    for line in json_lines:
        text = line['sentence_text']
        tokenized_text = keep_only_words(text.split())
        if tokenized_text:
            tokenized_sentences.append(tokenized_text)
    return tokenized_sentences

#A method that finds for each of the list words the most 5 similar words out of the list and write the results to a txt file (section 2.1)
def most_similar_words(model, words_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word1 in words_list:
            similarities = [(word2, model.wv.similarity(word1, word2)) for word2 in words_list if word1 != word2]
            similarities.sort(key=lambda x: x[1], reverse=True)
            written_text = [f"({word}, {similarity})" for word, similarity in similarities[:5]]
            file.write(f'{word1}: ')
            file.write(', '.join(written_text))
            file.write('\n')

def main():
    file = 'knesset_corpus.jsonl'
    
    # Extract the lines from the JSONL file (section 1.1)
    json_lines = json_lines_extract(file)
    #convert each line to a list of only words tokens (section 1.1)
    tokenized_sentences = json_lines_to_tokens(json_lines)

    #creating word2vec model (section 1.2)
    model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1)
    #saving the model (section 1.2)
    model.save("knesset_word2vec.model")
    #using the model and testing it (section 1.3)
    word_vectors = model.wv
    print(word_vectors['ישראל'])

    #finding the 5 most similar words to each of the list words and writing the results to the txt file (section 2.1)
    words_list = ['ישראל', 'גברת', 'ממשלה', 'חבר', 'בוקר', 'מים', 'אסור', 'רשות', 'זכויות']
    most_similar_words(model, words_list, 'knesset_similar_words.txt')


if __name__ == '__main__':
    main()


