import json
import random 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import sys
import os
from gensim.models import Word2Vec
random.seed(42)
np.random.seed(42)

class Sentence:
    def __init__(self, protocol_name, knesset, protocol_type, protocol_no, speaker, text):
        self.protocol_name = protocol_name
        self.knesset = knesset
        self.protocol_type = protocol_type
        self.protocol_no = protocol_no
        self.speaker = speaker
        self.text = text

class speaker:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences                

#A method that extracts the json lines from a JSONL file
def json_lines_extract(file):
    try: 
        with open(file, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        print('The file {file} is not found')
        sys.exit(1)

#A method extracts the top two speakers
def top_two_speakers(json_lines):
    speakers = {}
    for line in json_lines:
        try:
            speaker = line['speaker_name']
            if speaker in speakers:
                speakers[speaker] += 1
            else:
                speakers[speaker] = 1
        except KeyError:
            print('Skipping line due to missing speaker name')
            continue
    sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_speakers) < 2:
        raise ValueError("There's less than two speaker names in the data")
    return sorted_speakers[0][0], sorted_speakers[1][0] 

# A method that splits the sentences according to the speaker 
def split_data_by_speaker(json_lines, speaker1, speaker2):
    speaker1_data = []
    speaker2_data = []
    speaker1_splitted = speaker1.split()
    speaker2_splitted = speaker2.split()
    for line in json_lines:
        speaker = line['speaker_name']
        if speaker == speaker1:
            speaker1_data.append(line)
        elif speaker == speaker2:
            speaker2_data.append(line)
        else:
            speaker_splitted = speaker.split()
            
            #the case where the sentence said by one of the two speakers, but one of them is without first name 
            if (len(speaker_splitted) == 1 or len(speaker1_splitted) == 1) and speaker_splitted[-1] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif (len(speaker_splitted) == 1 or len(speaker2_splitted) == 1) and speaker_splitted[-1] == speaker2_splitted[-1]:
                speaker2_data.append(line)

            #the case where the sentence said by one of the two speakers, but one of the first names is abbreviated 
            elif len(speaker_splitted) > 1 and len(speaker1_splitted) > 1 and speaker_splitted[0][0] == speaker1_splitted[0][0] and (speaker_splitted[0][1] == "'" or speaker1_splitted[0][1] == "'") and speaker_splitted[-1] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif len(speaker_splitted) > 1 and len(speaker1_splitted) > 1 and speaker_splitted[0][0] == speaker2_splitted[0][0] and (speaker_splitted[0][1] == "'" or speaker2_splitted[0][1] == "'") and speaker_splitted[-1] == speaker2_splitted[-1]:
                speaker2_data.append(line)
            
            #the case where the sentence said by one of the two speakers, but one of the names has middle name
            elif len(speaker_splitted) > 1 and speaker_splitted[0] == speaker1_splitted[0] and speaker_splitted[-1] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif len(speaker_splitted) > 1 and speaker_splitted[0] == speaker2_splitted[0] and speaker_splitted[-1] == speaker2_splitted[-1]:
                speaker2_data.append(line)

            #special case where the one of the two speakers is Reuven Rivlin and its known nickname used as first name 
            elif ((len(speaker_splitted) > 1 and speaker_splitted[0] == 'רובי') or (len(speaker1_splitted) > 1 and speaker1_splitted[0] == 'רובי')) and speaker_splitted[-1] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif ((len(speaker_splitted) > 1 and speaker_splitted[0] == 'רובי') or (len(speaker2_splitted) > 1 and speaker2_splitted[0] == 'רובי')) and speaker_splitted[-1] == speaker2_splitted[-1]:
                speaker2_data.append(line) 
    return speaker1_data, speaker2_data

# A method that do down-sampling 
def down_sample(classes_data):
    class_count = min([len(class_data) for class_data in classes_data]) #finding the minimum class count
    downed_classes_data = []
    #extract sentences of each class as the min class count 
    for class_data in classes_data:
        downed_class_data = random.sample(class_data, class_count)
        downed_classes_data.append(downed_class_data)
    return downed_classes_data

# A method that trains and evaluates the classifier and returns the classification report
def classifier_evaluate(model, features_vectors, labels):
    preds = cross_val_predict(model, features_vectors, labels, cv=5)
    report = classification_report(labels, preds)
    return report
    
def main():
    #if len(sys.argv) != 3:
    #    print("3 arguments are required")
    #    sys.exit(1)
    
    #jsonl_file = sys.argv[1]
    #model = sys.argv[2]

    jsonl_file = 'knesset_corpus.jsonl'
    input_file = 'knesset_word2vec.model'

    #extracting the json lines from the JSONL file
    json_lines = json_lines_extract(jsonl_file)

    #extracting the top two speakers 
    speaker1, speaker2 = top_two_speakers(json_lines)

    #split the data according to the speaker
    first_full_data, second_full_data = split_data_by_speaker(json_lines, speaker1, speaker2)
    
    #class balancing - binary classification
    first_binary_data, second_binary_data = down_sample([first_full_data, second_full_data])

    #Creating the sentences and speakers objects 
    first_binary_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker1', line['sentence_text']) for line in first_binary_data]
    first_binary = speaker(speaker1, first_binary_sentences)
    second_binary_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker2', line['sentence_text']) for line in second_binary_data]
    second_binary = speaker(speaker2, second_binary_sentences)

    #Tf-idf vector creation - binary classificaion
    all_binary_sentences = first_binary_sentences + second_binary_sentences
    tfidf_binary_vectorizer, tfidf_binary_vectors = tfidf_vector_creator(all_binary_sentences)

    #Our vector creation - binary classificaion 
    features_binary_vectors = custom_vector_creator(all_binary_sentences, [first_binary, second_binary])

    #Labels for the vectors - binary classificaion 
    binary_labels = [line.speaker for line in all_binary_sentences]

    #initializing the binary classification classifiers
    knn_binary_tfidf = KNeighborsClassifier(n_neighbors=8, weights='distance')
    knn_binary_custom = KNeighborsClassifier(n_neighbors=8, p=1, weights='distance')

    #training the binary classification classifiers
    knn_binary_tfidf.fit(tfidf_binary_vectors, binary_labels)
    knn_binary_custom.fit(features_binary_vectors, binary_labels)

    #evaluating the binary classification classifiers
    for model in [(knn_binary_tfidf, tfidf_binary_vectors, 'KNN', 'tf-idf'), (knn_binary_custom, features_binary_vectors, 'KNN', 'custom')]:
        binary_report = classifier_evaluate(model[0], model[1], binary_labels)
        print(f'{model[2]} classifier with {model[3]} features:')
        print(binary_report)

if __name__ == '__main__':
    main()
