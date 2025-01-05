from gensim.models import Word2Vec
import json
import sys
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


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
        tokenized_sentences.append(tokenized_text)
    return tokenized_sentences

#A method that finds for each of the list words the most 5 similar words out of the list and write the results to a txt file (section 2.1)
def most_similar_words(word_vectors, words_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for word1 in words_list:
            similarities = [(word2, word_vectors.similarity(word1, word2)) for word2 in words_list if word1 != word2]
            similarities.sort(key=lambda x: x[1], reverse=True)
            written_text = [f"({word}, {similarity})" for word, similarity in similarities[:5]]
            file.write(f'{word1}: ')
            file.write(', '.join(written_text))
            file.write('\n')

#A method that creates embeddings for each of given sentences (section 2.2)
def sentences_embed(sentences, word_vectors):
    embedddings = []
    for sentence in sentences:
        sentence_embedding = [word_vectors[word] for word in sentence if word in word_vectors]
        if sentence_embedding:
            embedddings.append(np.mean(sentence_embedding, axis=0))
        else:
            embedddings.append(np.zeros(100))
    return embedddings

#A method that finds the most similar sentence for each of the given sentences and write the results to given txt file (section 2.3)
def most_similar_sentence(chosen_sentences, chosen_sentences_embeddings, json_lines, sentences_embeddings, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i in range(len(chosen_sentences)):
            similarities = cosine_similarity([chosen_sentences_embeddings[i]], sentences_embeddings)
            most_similar_index = similarities.argsort()[0][-2] #the 2nd most similar since the most similar is the sentence itself
            original_sentence = chosen_sentences[i]
            most_similar_sentence = json_lines[most_similar_index]['sentence_text']
            file.write(f"{original_sentence}: most similar sentence: {most_similar_sentence}\n")
        

def main():
    file = 'knesset_corpus.jsonl'
    
    # Extract the lines from the JSONL file (section 1.1)
    json_lines = json_lines_extract(file)
    #convert each line to a list of only words tokens (section 1.1)
    tokenized_sentences = json_lines_to_tokens(json_lines)

    #creating word2vec model (section 1.2)
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)
    #saving the model (section 1.2)
    model.save("knesset_word2vec.model")

    #using the model and testing it (section 1.3)
    word_vectors = model.wv
    #print(word_vectors['ישראל'])

    #finding the 5 most similar words to each of the list words and writing the results to the txt file (section 2.1)
    words_list = ['ישראל', 'גברת', 'ממשלה', 'חבר', 'בוקר', 'מים', 'אסור', 'רשות', 'זכויות']
    most_similar_words(word_vectors, words_list, 'knesset_similar_words.txt')

    #creating sentene embeddings for each sentence in the corpus (section 2.2)
    sentences_embeddings = sentences_embed(tokenized_sentences, word_vectors)

    #finding the most similar sentence for a collection of 10 sentences (section 2.3)
    chosen_sentences = ['אמרו שזכות עם ישראל להתנחל ביהודה ושומרון , ועל - פי המדיניות הזאת של ממשלת ישראל כל מי שהלך להתנחל בשטחי יש"ע פעל על - פי ההוראה הזאת ודומות לה .', 
                        "יש מדד חדש שנקרא מדד הפריפריאליות שמשרד האוצר והלשכה המרכזית לסטטיסטיקה פרסמו לפני מספר חודשים ויש לזה ביטוי בהחלטות הממשלה האחרונות לקראת תקציב 2009 .", 
                        "מדינת ישראל תהיה הריבונית מהירדן ועד לים .", 
                        "דחייה כזו או אחרת במתן שירותים במשרד הבריאות יש לה משמעות אחת , לפגיעה בתקציבו : סבל לחולים , פגיעה פיזית בנזקקים ואולי אפילו מותם של אחדים , כי במי אנו מדברים ?", 
                        "המוסד לביטוח לאומי הוא המוסמך , על - פי חוק , לקבוע מהי ההכשרה המתאימה לשיקומם של נכים .", 
                        "אמרתי שנמתין , נחכה ונראה מה יש לו להגיד בעניין .", 
                        "תודה רבה חבר הכנסת כבל .", 
                        "זה כואב , זה מרגיז וצריך למצוא פתרונות .", 
                        "אני מאוד שמחה , אבל אני גם מאוד מתרגשת , עודד .", 
                        "המתווה הזה הוא טוב ."]
    tokenized_chosen_sentences = [keep_only_words(sentence.split()) for sentence in chosen_sentences]
    chosen_sentences_embeddings = sentences_embed(tokenized_chosen_sentences, word_vectors) #creating embeddings for the chosen sentences
    most_similar_sentence(chosen_sentences, chosen_sentences_embeddings, json_lines, sentences_embeddings, 'knesset_similar_sentences.txt')


if __name__ == '__main__':
    main()


