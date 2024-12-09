import json
import math 
import sys
import pandas as pd

class Trigram_LM:
    def __init__(self, sentences):
        self.sentences = [['s_0', 's_1'] + sentence.split() for sentence in sentences]
        self.unique_tokens_count = 0
        self.tokens_count = 0
        self.trigrams = self._trigrams_create(self.sentences)
        self.bigrams = self._bigrams_create(self.sentences)
        self.unigrams = self._unigrams_create(self.sentences)
    def _trigrams_create(self, sentences):
        trigrams_dict = {}
        for sentence in sentences:
            for i,token in enumerate(sentence[2:]):
                if (sentence[i-2], sentence[i-1]) not in trigrams_dict:
                    trigrams_dict[(sentence[i-2], sentence[i-1])] = {token : 1}
                elif token not in trigrams_dict[(sentence[i-2], sentence[i-1])]:
                    trigrams_dict[(sentence[i-2], sentence[i-1])][token] = 1
                else:
                    trigrams_dict[(sentence[i-2], sentence[i-1])][token] += 1
        return trigrams_dict
    def _bigrams_create(self, sentences):
        bigrams_dict = {}
        for sentence in sentences: 
            for i,token in enumerate(sentence[1:]):
                if (sentence[i-1]) not in bigrams_dict:
                    bigrams_dict[(sentence[i-1])] = {token : 1}
                elif token not in bigrams_dict[(sentence[i-1])]:
                    bigrams_dict[(sentence[i-1])][token] = 1
                else:
                    bigrams_dict[(sentence[i-1])][token] += 1
        return bigrams_dict
    def _unigrams_create(self, sentences):
        unigrams_dict = {}
        for sentence in sentences:
            for token in sentence:
                self.tokens_count += 1
                if token not in unigrams_dict:
                    unigrams_dict[token] = 1
                else:
                    unigrams_dict[token] += 1
        self.unique_tokens_count = len(unigrams_dict.keys())
        return unigrams_dict
    
    def calculate_prob_of_sentence(self, s):
        s = ['s_0', 's_1'] + s.split()
        total_prob = 1
        for i in range(2, len(s)):
            prob = 0
            denominator = numerator = 0
            #computing the probability of the trigram s[i-2] s[i-1] s[i]
            if (s[i-2], s[i-1]) in self.trigrams:
                denominator = sum(self.trigrams[(s[i-2], s[i-1])].values()) + self.unique_tokens_count
            else:
                denominator = self.unique_tokens_count
            if (s[i-2], s[i-1]) in self.trigrams and s[i] in self.trigrams[(s[i-2], s[i-1])]: 
                numerator = self.trigrams[(s[i-2], s[i-1])][s[i]] + 1
            else:
                numerator = 1
            prob += 0.6 * numerator / denominator
            #computing the probability of the bigrams s[i-1] s[i]
            if (s[i-1]) in self.bigrams:
                denominator = sum(self.bigrams[(s[i-1])].values()) + self.unique_tokens_count
            else:
                denominator = self.unique_tokens_count
            if (s[i-1]) in self.bigrams and s[i] in self.bigrams[(s[i-1])]:
                numerator = self.bigrams[(s[i-1])][s[i]] + 1
            else:
                numerator = 1
            prob += 0.3 * numerator / denominator
            #computing the probability of s[i]
            numerator = self.unigrams[s[i]] + 1 if s[i] in self.unigrams else 1
            prob += 0.1 * numerator / (self.tokens_count + self.unique_tokens_count)
            total_prob *= prob
        return total_prob
    def generate_next_token(self, s):
        s = s.split()
        next_token = None
        next_token_prob = -1
        if len(s) == 0 and ('s_0', 's_1') in self.trigrams:
            for token in self.trigrams[('s_0', 's_1')]:
                    token_prob = self.calculate_prob_of_sentence(token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token
        elif len(s) == 1:
            if ('s_1', s[-1]) in self.trigrams:
                for token in self.trigrams[('s_1', s[-1])]:
                    token_prob = self.calculate_prob_of_sentence(s[-1] + " " + token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token 
            elif (s[-1]) in self.bigrams:
                for token in self.bigrams[(s[-1])]:
                    token_prob = self.calculate_prob_of_sentence(s[-1] + " " + token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token   
            else:
                for token in self.unigrams:
                    token_prob = self.calculate_prob_of_sentence(token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token     
        else:
            if (s[-2], s[-1]) in self.trigrams:
                for token in self.trigrams[(s[-2], s[-1])]:
                    token_prob = self.calculate_prob_of_sentence(s[-2] + " " + s[-1] + " " + token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token 
            elif (s[-1]) in self.bigrams:
                for token in self.bigrams[(s[-1])]:
                    token_prob = self.calculate_prob_of_sentence(s[-1] + " " + token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token 
            else:
                for token in self.unigrams:
                    token_prob = self.calculate_prob_of_sentence(token)
                    if token_prob > next_token_prob:
                        next_token_prob = token_prob
                        next_token = token 
        return (next_token, next_token_prob)

def get_k_n_t_collocations(k, n, t, corpus, type):
    collocations = {}
    protocols = {}
    #extracting all the collocations of long n
    for idx, row in corpus.iterrows():
        sentence_splitted = row['sentence_text'].split()
        protocol = row['protocol_number']
        for i in range(n, len(sentence_splitted)):
            collocation = tuple(sentence_splitted[i-n : i])
            #handling the number of appereance of the collocation by protocol 
            if collocation in collocations:
                if protocol in collocations[collocation]:
                    collocations[collocation][protocol] += 1
                else:
                    collocations[collocation][protocol] = 1
            else:
                collocations[collocation] = {protocol : 1}
            #handling the number of collocation in each protocol 
            if protocol in protocols:
                if collocation in protocols[protocol]:
                    protocols[protocol][collocation] += 1
                else:
                    protocols[protocol][collocation] = 1
            else:
                protocols[protocol] = {collocation : 1}
    #keeping only the collocations that appears at lest t times 
    relevant_collocations = {}
    for collocation in collocations:
        if sum(collocations[collocation].values()) >= t:
            relevant_collocations[collocation] = collocations[collocation] 
    #sorting the collocations according to the criterion
    sorted_collocations = []
    if type == 'frequency': 
        sorted_collocations = [collocation for collocation, _ in sorted(relevant_collocations.items(), key=lambda x: sum(x[1].values()), reverse=True)]
    elif type == 'tfidf':
        collocations_by_tfidf = {}
        for collocation in relevant_collocations:
            collocations_by_tfidf[collocation] = 0 
            for protocol in collocations[collocation]:
                tf = collocations[collocation][protocol] / sum(protocols[protocol].values())
                idf = math.log(len(protocols.keys()) / len(collocations[collocation].keys()))
                collocations_by_tfidf[collocation] += tf * idf 
            collocations_by_tfidf[collocation] /= len(protocols.keys())
        sorted_collocations = [collocation for collocation, _ in sorted(collocations_by_tfidf.items(), key=lambda x: x[1], reverse=True)]
    #choosing the k most common collocations and joining them 
    collocations_to_return = []
    for collocation in sorted_collocations[:min(k, len(sorted_collocations))]:
        joined_collocation = ' '.join(collocation)
        collocations_to_return.append(joined_collocation)
    return collocations_to_return

def main():
    #if len(sys.argv) != 3:
    #    print("Error: Incorrect # of arguments.\n")
    #    sys.exit(1)
    #else:
    #    print("Creating the output ..\n")
    #file_path = sys.argv[1] 
    file_path = 'knesset_corpus.json'
    #output_folder = sys.argv[2] 
    output_folder = 'output'
    lines_list = None
    with open(file_path, "r", encoding = "utf-8") as file:
        lines_list =  [json.loads(line) for line in file]
    #Separating the committee and plenary data
    committee_sentences = [line["sentence_text"] for line in lines_list if line["protocol_type"] == 'committee']
    plenary_sentences = [line["sentence_text"] for line in lines_list if line["protocol_type"] == 'plenary']
    committee_model = Trigram_LM(committee_sentences)
    plenary_model = Trigram_LM(plenary_sentences)
    corpus_df = pd.DataFrame(lines_list)
    committee_df = corpus_df[corpus_df['protocol_type'] == 'committee']
    plenary_df = corpus_df[corpus_df['protocol_type'] == 'plenary']
    #
    output = 'knesset_collocations.txt'
    with open(output, 'w', encoding = 'utf-8') as file:
        for n in [2,3,4]:
            file.write(f"{n}-gram collocations:\n")
            for type in ['frequency', 'tfidf']:
                file.write(f"{type.capitalize()}:\n")
                for corpus_name, corpus_df in [("Committee corpus", committee_df), ("Plenary corpus", plenary_df)]:
                    file.write(f"{corpus_name}:\n")
                    collocations = get_k_n_t_collocations(10, n, 5, corpus_df, type)
                    for collocation in collocations:
                        file.write(f"{collocation}\n")
                    file.write("\n")
            file.write("\n")

    #checks 
    sentence_prob = committee_model.calculate_prob_of_sentence('אחמד טיבי')
    print(sentence_prob)
    next_word = committee_model.generate_next_token('אחמד')
    print(next_word)
    relevant_sentences = get_k_n_t_collocations(5, 5, 5, committee_df, 'frequency')
    for i in range(len(relevant_sentences)):
        print (relevant_sentences[i])

if __name__ == "__main__":
    main()