import json
import random 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import sys
import os
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
        self.n_grams = {}
        for sentence in sentences:
            words = sentence.text.split()
            for i in range(len(words)):
                if words[i] in self.n_grams:
                    self.n_grams[words[i]] += 1
                else:
                    self.n_grams[words[i]] = 1
                if i < len(words) - 1:
                    if (words[i], words[i+1]) in self.n_grams:
                        self.n_grams[(words[i], words[i+1])] += 1
                    else:
                        self.n_grams[(words[i], words[i+1])] = 1
                if i < len(words) - 2:
                    if (words[i], words[i+1], words[i+2]) in self.n_grams:
                        self.n_grams[(words[i], words[i+1], words[i+2])] += 1
                    else:
                        self.n_grams[(words[i], words[i+1], words[i+2])] = 1
                
                

#A method that extracts the json lines from a JSONL file (section 1)
def json_lines_extract(file):
    try: 
        with open(file, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        print('The file {file} is not found')
        sys.exit(1)

#A method extracts the top two speakers (section 1)
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
    #print(sorted_speakers[:2]) #used for printing the top two speakers
    if len(sorted_speakers) < 2:
        raise ValueError("There's less than two speaker names in the data")
    return sorted_speakers[0][0], sorted_speakers[1][0] 

# A method that splits the sentences according to the speaker (section 1.2)
def split_data_by_speaker(json_lines, speaker1, speaker2):
    speaker1_data = []
    speaker2_data = []
    other_data = []
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

            #the case where the sentence not said by one of the two speakers
            else:
                other_data.append(line)
    return speaker1_data, speaker2_data, other_data

# A method that do down-sampling (section 2)
def down_sample(classes_data):
    class_count = min([len(class_data) for class_data in classes_data]) #finding the minimum class count
    downed_classes_data = []
    #extract sentences of each class as the min class count 
    for class_data in classes_data:
        downed_class_data = random.sample(class_data, class_count)
        downed_classes_data.append(downed_class_data)
    return downed_classes_data

# A method that creates tf-idf vectors (section 3.1)
def tfidf_vector_creator(lines):
    all_texts = [line.text for line in lines]
    vectorizer = TfidfVectorizer()
    tfidfVectors = vectorizer.fit_transform(all_texts)
    return vectorizer, tfidfVectors

#A helper method that finds the collocations with the highest difference in appeareance between each two speakers
def decisive_collocations(speakers):
    highest_diff_collocations = set()
    for first_speaker in speakers:
        for second_speaker in [speaker for speaker in speakers if speaker != first_speaker]:
            high_diff_collocations = {} #will store the collocations with the highest difference in appeareance between the two speakers
            for collocation in first_speaker.n_grams:
                if collocation in second_speaker.n_grams and first_speaker.n_grams[collocation] > second_speaker.n_grams[collocation]:
                    high_diff_collocations[collocation] = first_speaker.n_grams[collocation] - second_speaker.n_grams[collocation]
                elif collocation not in second_speaker.n_grams:
                    high_diff_collocations[collocation] = first_speaker.n_grams[collocation] 
            high_diff_collocations = sorted(high_diff_collocations.items(), key=lambda x: x[1], reverse=True)[:6]
            high_diff_collocations = [' '.join(collocation[0]) if isinstance(collocation[0], tuple) else collocation[0] for collocation in high_diff_collocations]  
            highest_diff_collocations.update(high_diff_collocations)
    return highest_diff_collocations

# A method that creates custom vectors of features (section 3.2)
def custom_vector_creator(lines, speakers):
    #finding the decisive collocations that will be relevant to the classification
    highest_diff_collocations = decisive_collocations(speakers)
    #creating the vectors
    vectors = []
    for line in lines:
        features_vector = []
        sentence_text = line.text
        sentence_splitted = line.text.split()

        #Feature 1: Knesset number 
        knesset_number = line.knesset
        features_vector.append(knesset_number)

        #Feature 2: Protocol Number
        protocol_number = line.protocol_no
        features_vector.append(protocol_number)
        
        #Feature 3: Protocol type
        protocol_type = 1 if line.protocol_type == 'committee' else 0
        features_vector.append(protocol_type)

        #Feature 4: Sentence length
        sentence_length = len(sentence_splitted)
        features_vector.append(sentence_length)

        #Feature 5: If digit appears in the sentence 
        contain_digit = 1 if any(token.isdigit() for token in sentence_splitted) else 0
        features_vector.append(contain_digit)
    
        #Collocation-related features: if one of the collocations below appears in the sentence
        for collocation in highest_diff_collocations:
            feature_value = sentence_text.count(collocation)
            features_vector.append(feature_value)
        vectors.append(features_vector)
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(vectors)
    return normalized_vectors

# A method that trains and evaluates the classifier and returns the classification report (section 4)
def classifier_evaluate(model, features_vectors, labels):
    preds = cross_val_predict(model, features_vectors, labels, cv=5)
    report = classification_report(labels, preds)
    return report

#A method that classifies sentences in input file to one of two speakers or other (section 5)
def sentences_classify(model, vectorizer, speaker1, speaker2, input_file, output_path):
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    features = vectorizer.transform(sentences)
    preds = model.predict(features)
    mapped_labels = {speaker1: "first", speaker2: "second", "other": "other"}
    classifications = [mapped_labels[str(pred)] for pred in preds]
    output_file = os.path.join(output_path, 'classification_results.txt')
    with open(output_file, 'w', encoding='utf-8') as file:
        for classification in classifications:
            file.write(classification + '\n') 
    
def main():
    if len(sys.argv) != 4:
        print("4 arguments are required")
        sys.exit(1)
    
    file = sys.argv[1]
    input_file = sys.argv[2]
    output_path = sys.argv[3]

    #file = 'knesset_corpus.jsonl'
    #input_file = 'knesset_sentences.txt'
    #output_path = 'output'
    os.makedirs(output_path, exist_ok=True)

    #extracting the json lines from the JSONL file
    json_lines = json_lines_extract(file)

    #extracting the top two speakers (section 1)
    speaker1, speaker2 = top_two_speakers(json_lines)

    #split the data according to the speaker (section 1)
    first_full_data, second_full_data, other_full_data = split_data_by_speaker(json_lines, speaker1, speaker2)
    
    #class balancing - binary classification (section 2)
    first_binary_data, second_binary_data = down_sample([first_full_data, second_full_data])

    #Creating the sentences and speakers objects 
    first_binary_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker1', line['sentence_text']) for line in first_binary_data]
    first_binary = speaker(speaker1, first_binary_sentences)
    second_binary_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker2', line['sentence_text']) for line in second_binary_data]
    second_binary = speaker(speaker2, second_binary_sentences)

    #classes balancing - multi-class classification (section 2)
    first_multi_data, second_multi_data, other_multi_data = down_sample([first_full_data, second_full_data, other_full_data])

    #Creating the sentences and speakers objects 
    first_multi_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker1', line['sentence_text']) for line in first_multi_data]
    first_multi = speaker(speaker1, first_multi_sentences)
    second_multi_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'speaker2', line['sentence_text']) for line in second_multi_data]
    second_multi = speaker(speaker2, second_multi_sentences)
    other_multi_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'other', line['sentence_text']) for line in other_multi_data]
    other_multi = speaker("other", other_multi_sentences)

    #printing the count of sentences of each binary classification class before and after the down sampling (section 2)
    # print('Sentences count of each binary classification class before the down sampling:')
    # print('first_sentences:', len(first_full_data))
    # print('second_sentences:', len(second_full_data))
    # print('Sentences count of each binary classification class after the down sampling:')
    # print('first_sentences:', len(first_binary_sentences))
    # print('second_sentences:', len(second_binary_sentences))

    #printing the count of sentences of each multiclass classification class before and after the down sampling (section 2)
    # print('Sentences count of each multiclass classification class before the down sampling:')
    # print('first_sentences:', len(first_full_data))
    # print('second_sentences:', len(second_full_data))
    # print('other_sentences:', len(other_full_data))
    # print('Sentences count of each multiclass classification class after the down sampling:')
    # print('first_sentences:', len(first_multi_sentences))
    # print('second_sentences:', len(second_multi_sentences))
    # print('other_sentences:', len(other_multi_sentences))

    #Tf-idf vector creation - binary classificaion (section 3.1)
    all_binary_sentences = first_binary_sentences + second_binary_sentences
    tfidf_binary_vectorizer, tfidf_binary_vectors = tfidf_vector_creator(all_binary_sentences)

    #Our vector creation - binary classificaion (section 3.2)
    features_binary_vectors = custom_vector_creator(all_binary_sentences, [first_binary, second_binary])

    #Labels for the vectors - binary classificaion (section 3.2)
    binary_labels = [line.speaker for line in all_binary_sentences]

    #Tf-idf vector creation - multiclass classificaion (section 3.1)
    all_multi_sentences = first_multi_sentences + second_multi_sentences + other_multi_sentences
    tfidf_multi_vectorizer, tfidf_multi_vectors = tfidf_vector_creator(all_multi_sentences)

    #Our vector creation - multiclass classificaion (section 3.2)        
    features_multi_vectors = custom_vector_creator(all_multi_sentences, [first_multi, second_multi, other_multi])

    #Labels for the vectors - multiclass classificaion (section 3.2)
    multi_labels = [line.speaker for line in all_multi_sentences]

    #initializing the binary classification classifiers (section 4)
    knn_binary_tfidf = KNeighborsClassifier(n_neighbors=8, weights='distance')
    logistic_reg_binary_tfidf = LogisticRegression(max_iter=1500)
    knn_binary_custom = KNeighborsClassifier(n_neighbors=8, p=1, weights='distance')
    logistic_reg_binary_custom = LogisticRegression(max_iter=1500,penalty='l1', solver='liblinear', C=1.0)

    #training the binary classification classifiers (section 4)
    knn_binary_tfidf.fit(tfidf_binary_vectors, binary_labels)
    logistic_reg_binary_tfidf.fit(tfidf_binary_vectors, binary_labels)
    knn_binary_custom.fit(features_binary_vectors, binary_labels)
    logistic_reg_binary_custom.fit(features_binary_vectors, binary_labels)

    #evaluating the binary classification classifiers (section 4)
    for model in [(knn_binary_tfidf, tfidf_binary_vectors, 'KNN', 'tf-idf'), (logistic_reg_binary_tfidf, tfidf_binary_vectors, 'Logistic Regression', 'tf-idf'), (knn_binary_custom, features_binary_vectors, 'KNN', 'custom'), (logistic_reg_binary_custom, features_binary_vectors, 'Logistic Regression', 'custom')]:
        binary_report = classifier_evaluate(model[0], model[1], binary_labels)
        #print(f'{model[2]} classifier with {model[3]} features:')
        #print(binary_report)

    #initializing the multiclass classification classifiers (section 4)
    knn_multi_tfidf = KNeighborsClassifier(n_neighbors=8, weights='distance')
    logistic_reg_multi_tfidf = LogisticRegression(max_iter=1500)
    knn_multi_custom = KNeighborsClassifier(n_neighbors=8, p=1, weights='distance')
    logistic_reg_multi_custom = LogisticRegression(max_iter=1500,penalty='l1', solver='liblinear', C=1.0)


    #training the multiclass classification classifiers (section 4)
    knn_multi_tfidf.fit(tfidf_multi_vectors, multi_labels)
    logistic_reg_multi_tfidf.fit(tfidf_multi_vectors, multi_labels)
    knn_multi_custom.fit(features_multi_vectors, multi_labels)
    logistic_reg_multi_custom.fit(features_multi_vectors, multi_labels)


    #evaluating the multiclass classification classifiers (section 4)
    for model in [(knn_multi_tfidf, tfidf_multi_vectors, 'KNN', 'tf-idf'), (logistic_reg_multi_tfidf, tfidf_multi_vectors, 'Logistic Regression', 'tf-idf'), (knn_multi_custom, features_multi_vectors, 'KNN', 'custom'), (logistic_reg_multi_custom, features_multi_vectors, 'Logistic Regression', 'custom')]:
        multi_report = classifier_evaluate(model[0], model[1], multi_labels)
        #print(f'{model[2]} classifier with {model[3]} features:')
        #print(multi_report)
    
    #Classifying the sentences in the input file (section 5)
    sentences_classify(logistic_reg_multi_tfidf, tfidf_multi_vectorizer, 'speaker1', 'speaker2', input_file, output_path)

if __name__ == '__main__':
    main()
