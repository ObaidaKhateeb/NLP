import json
import random 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
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

#A method that extracts the json lines from a JSONL file (section 1)
def json_lines_extract(file):
    with open(file, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

#A method extracts the top two speakers (section 1)
def top_two_speakers(json_lines):
    speakers = {}
    for line in json_lines:
        speaker = line['speaker_name']
        if speaker in speakers:
            speakers[speaker] += 1
        else:
            speakers[speaker] = 1
    sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
    #print(sorted_speakers[:2])
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
            
            #checking if the sentence said by one of the two speakers but with a different given name
            if len(speaker_splitted) == 1 and speaker_splitted[0] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif len(speaker_splitted) == 1 and speaker_splitted[0] == speaker2_splitted[-1]:
                    speaker2_data.append(line)
            elif len(speaker_splitted) > 1 and  speaker_splitted[0][0] == speaker1_splitted[0][0] and speaker_splitted[-1] == speaker1_splitted[-1]:
                speaker1_data.append(line)
            elif len(speaker_splitted) > 1 and speaker_splitted[0][0] == speaker2_splitted[0][0] and speaker_splitted[-1] == speaker2_splitted[-1]:
                    speaker2_data.append(line)
            
            #the case where the sentence supposed to be said by someone other than the two
            else:
                other_data.append(line)
    return speaker1_data, speaker2_data, other_data

# A method that creates tf-idf vectors (section 3.1)
def tfidf_vector_creator(lines):
    all_texts = [line.text for line in lines]
    vectorizer = TfidfVectorizer()
    tfidfVectors = vectorizer.fit_transform(all_texts)
    return vectorizer, tfidfVectors

# A method that creates vector of features (section 3.2)
def our_vector_creator(lines):
    all_texts = [line.text for line in lines]
    features = []
    for line in lines:
        features_vector = []
        sentence_splitted = line.text.split()

        #Feature 1: Knesset number 
        knesset_number = line.knesset
        features_vector.append(knesset_number)
        
        #Feature 2: Protocol type
        feature_value = 1 if line.protocol_type == 'committee' else 0
        features_vector.append(feature_value)

        #Feature 3: Sentence length
        sentence_length = len(sentence_splitted)
        features_vector.append(sentence_length)

        #Feature 4: If digit appears in the sentence 
        feature_value = 1 if any(token.isdigit() for token in sentence_splitted) else 0
        features_vector.append(feature_value)

        #Rest of features: if one of the collocations below appears in the sentence
        collocations = [ 'אני', 'חבר הכנסת', 'חברי הכנסת', 'לחבר הכנסת', 'הצעת חוק', 'רבותי חברי', 'כהצעת הוועדה', 'ההסתייגות של', 'אדוני היושב', 'רבותי חברי הכנסת', 'בבקשה', 'תודה' ]
        for collocation in collocations:
            feature_value = 1 if collocation in sentence_splitted else 0
            features_vector.append(feature_value)
        features.append(features_vector)
    return features

# A method that trains and evaluates the classifier and returns the classification report (section 4)
def classifier_evaluate(model, features_vectors, labels):
    preds = cross_val_predict(model, features_vectors, labels, cv=5)
    report = classification_report(labels, preds)
    return report

def sentences_classify(model, vectorizer, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    features = vectorizer.transform(sentences)
    preds = model.predict(features)
    mapped_labels = {"ראובן ריבלין": "first", "א' בורג": "second", "Other": "other"}
    classifications = [mapped_labels.get(pred, "other") for pred in preds]
    with open(output_file, 'w', encoding='utf-8') as file:
        for classification in classifications:
            file.write(classification + '\n') 
    
def main():
    file = 'knesset_corpus.jsonl'
    json_lines = json_lines_extract(file)

    #extracting the top two speakers in section 1
    speaker1, speaker2 = top_two_speakers(json_lines)

    #split the data according to the speaker (section 1.2)
    first_full_data, second_full_data, other_full_data = split_data_by_speaker(json_lines, speaker1, speaker2)
    
    #classes balancing (section 2)
    class_count = min(len(first_full_data), len(second_full_data), len(other_full_data))
    first_data = random.sample(first_full_data, class_count)
    second_data = random.sample(second_full_data, class_count)
    other_data = random.sample(other_full_data, class_count)

    # #printing the count of sentences of each class before and after the down sampling (section 2)
    # print('Sentences count of each class before the down sampling:')
    # print('first_sentences:', len(first_full_data))
    # print('second_sentences:', len(second_full_data))
    # print('other_sentences:', len(other_full_data))
    # print('Sentences count of each class after the down sampling:')
    # print('first_sentences:', len(first_data))
    # print('second_sentences:', len(second_data))
    # print('other_sentences:', len(other_data))


    #Creating the sentences and speakers objects (pre-section 3)
    first_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'Rivlin', line['sentence_text']) for line in first_data]
    first = speaker(speaker1, first_sentences)
    second_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'Burg', line['sentence_text']) for line in second_data]
    second = speaker(speaker2, second_sentences)
    other_sentences = [Sentence(line['protocol_name'], line['knesset_number'], line['protocol_type'], line['protocol_number'], 'Other', line['sentence_text']) for line in other_data]
    other = speaker("other", other_sentences)

    #Tf-idf vector creation (section 3.1)
    all_sentences = first_sentences + second_sentences + other_sentences
    tfidf_vectorizer, tfidf_vectors = tfidf_vector_creator(all_sentences)

    #Our vector creation (section 3.2)
    features_vectors = our_vector_creator(all_sentences)

    #Labels for the vectors (section 3.2)
    labels = [line.speaker for line in all_sentences]

    #initializing the classifiers (section 4)
    knn_tfidf = KNeighborsClassifier(n_neighbors=5)
    logistic_reg_tfidf = LogisticRegression(max_iter=1000)
    knn_custom = KNeighborsClassifier(n_neighbors=5)
    logistic_reg_custom = LogisticRegression(max_iter=1000)


    #training the classifiers (section 4)
    knn_tfidf.fit(tfidf_vectors, labels)
    logistic_reg_tfidf.fit(tfidf_vectors, labels)
    knn_custom.fit(features_vectors, labels)
    logistic_reg_custom.fit(features_vectors, labels)


    #evaluating the classifiers (section 4)
    for model in [(knn_tfidf, tfidf_vectors, 'KNN', 'tf-idf'), (logistic_reg_tfidf, tfidf_vectors, 'Logistic Regression', 'tf-idf'), (knn_custom, features_vectors, 'KNN', 'custom'), (logistic_reg_custom, features_vectors, 'Logistic Regression', 'custom')]:
            report = classifier_evaluate(model[0], model[1], labels)
            print(f'{model[2]} classifier with {model[3]} features:')
            print(report)
    
    #Classifying the sentences in the input file (section 5)
    sentences_classify(logistic_reg_tfidf, tfidf_vectorizer, 'knesset_sentences.txt', 'classification_results.txt')

if __name__ == '__main__':
    main()