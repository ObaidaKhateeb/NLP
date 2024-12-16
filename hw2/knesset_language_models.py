import math 
import pandas as pd
import random
import os

class Trigram_LM:
    def __init__(self, sentences):
        self.sentences = [['s_0', 's_1'] + sentence.split() for sentence in sentences]
        self.unique_tokens_count = 0
        self.tokens_count = 0
        self.trigrams = self._trigrams_create(self.sentences)
        self.bigrams = self._bigrams_create(self.sentences)
        self.unigrams = self._unigrams_create(self.sentences)

    #A method that creates a dictionary of trigrams frequencies out of the given sentences. 
    #Input: List of tokenized sentences. 
    #Output: Dictionary of trigrams, each with its frequency. 
    def _trigrams_create(self, sentences):
        trigrams_dict = {}
        for sentence in sentences:
            for i,token in enumerate(sentence[2:]):
                if (sentence[i-2], sentence[i-1], sentence[i]) in trigrams_dict:
                    trigrams_dict[(sentence[i-2], sentence[i-1], sentence[i])] += 1
                else:
                    trigrams_dict[(sentence[i-2], sentence[i-1], sentence[i])] = 1
        return trigrams_dict
    
    #A method that creates a dictionary of bigrams frequencies out of the given sentences. 
    #Input: List of tokenized sentences. 
    #Output: Dictionary of bigrams, each with its frequency. 
    def _bigrams_create(self, sentences):
        bigrams_dict = {}
        for sentence in sentences: 
            for i,token in enumerate(sentence[1:]):
                if (sentence[i-1], sentence[i]) in bigrams_dict:
                    bigrams_dict[(sentence[i-1], sentence[i])] += 1
                else:
                    bigrams_dict[(sentence[i-1], sentence[i])] = 1
        return bigrams_dict
    
    #A method that creates a dictionary of unigrams frequencies out of the given sentences. In addition, it computes tokens count and tokens unique count. 
    #Input: List of tokenized sentences. 
    #Output: Dictionary of unigrams, each with its frequency. 
    def _unigrams_create(self, sentences):
        unigrams_dict = {}
        for sentence in sentences:
            for i, token in enumerate(sentence[1:]):
                if token in unigrams_dict:
                    unigrams_dict[token] += 1
                else:
                    unigrams_dict[token] = 1
                if i not in [0,1]: #counts the total number of tokens, excepting 's_0' and 's_1'
                    self.tokens_count += 1
        self.unique_tokens_count = len(unigrams_dict.keys())
        if len(unigrams_dict.keys()): #substracting the 's_0' and 's_1' tokens, the 'if' condition added to deal with the edge case where the sentences list is empty
            self.unique_tokens_count -= 2 
        return unigrams_dict

    #A helper method for 'calculate_prob_of_sentence'. It computes the log probability of a token based on the trigram formula.
    #Input: 3-long sentence, in which the intended token is the last token. 
    #Output: log probability of the token. 
    def calculate_prob_of_token(self, s):
        if self.unique_tokens_count == 0:
            raise ValueError('Error: The model has no tokens')
        lambda1, lambda2, lambda3 = 0.85, 0.14, 0.01
        trigram = (s[0], s[1], s[2])
        bigram = (s[0], s[1])
        unigram = (s[0])
        #computing the probability of the trigram s[i-2] s[i-1] s[i]
        numerator = self.trigrams.get(trigram, 0) + 1
        denominator = self.bigrams.get(bigram, 0) + self.unique_tokens_count
        trigram = numerator / denominator
        #computing the probability of the bigrams s[i-1] s[i]
        numerator = self.bigrams.get(bigram, 0) + 1
        denominator = self.unigrams.get(unigram, 0) + self.unique_tokens_count
        bigram = numerator / denominator
        #computing the probability of the unigram s[i]
        numerator = self.unigrams.get(unigram, 0) + 1
        denominator = self.tokens_count + self.unique_tokens_count
        unigram = numerator / denominator
        prob = lambda1 * trigram + lambda2 * bigram + lambda3 * unigram
        return math.log(prob)

    #A method that computes the log probability of a sentence based on the trigram formula which takes into account trigrams, bigrams, and unigrams.
    #Input: Sentence. 
    #Output: log probability of the sentence. 
    def calculate_prob_of_sentence(self, s):
        s = ['s_0', 's_1'] + s.split()
        total_prob = 0
        for i in range(2, len(s)):
            total_prob += self.calculate_prob_of_token(s[i-2:i+1])
        return total_prob
    
    #A method that predict the token with the highest probability to be the next of a given sentence.
    #Input: sentence.
    #Output: A tuple of 2 elements: the predicted token and its probability. 
    def generate_next_token(self, s):
        s = s.split()
        next_token = None
        next_token_prob = float('-inf')
        for token in (key for key in self.unigrams if key != 's_1'): #iterating over the different tokens, excepting 's_1'
            s_new = ' '.join(s) + ' ' + token #add the token to the end of the sentence 
            try:
                token_prob = self.calculate_prob_of_sentence(s_new) #computing the probability of the sentence with the added token
            except ValueError as e:
                print(f'Error: Failed to calculate probability of {token}: {e}')
                continue
            if token_prob > next_token_prob:
                next_token_prob = token_prob
                next_token = token
        return (next_token, next_token_prob) #exp of log of the probability will return the probability itself as required 

#Extracts and return collection of collocations based on a given parameters. 
#Input: number of collocations to return - k, length of the collocations - n, min number of occurence for a collocation - t, dataframe of corpus, ranking criterion - type
#Output: List of k most common collocations of length n, that occured at least t times in the given corpus, based on the given type. 
def get_k_n_t_collocations(k, n, t, corpus, type):
    collocations = {}
    protocols = {}
    #extracting all the collocations of long n
    for idx, row in corpus.iterrows():
        sentence_splitted = row['sentence_text'].split()
        protocol = row['protocol_number']
        if protocol not in protocols:
            protocols[protocol] = {}
        for i in range(n, len(sentence_splitted)): #sentences shorter than n will not enter the for loop
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
            if collocation in protocols[protocol]:
                protocols[protocol][collocation] += 1
            else:
                protocols[protocol][collocation] = 1
    #keeping only the collocations that appears at lest t times 
    relevant_collocations = {}
    for collocation in collocations:
        if sum(collocations[collocation].values()) >= t: #the sum over the different protocols >= t 
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
                tf = collocations[collocation][protocol] / sum(protocols[protocol].values()) #computing tf  
                idf = math.log2(len(protocols.keys()) / len(collocations[collocation].keys())) #computing idf
                collocations_by_tfidf[collocation] += tf * idf 
            collocations_by_tfidf[collocation] /= len(protocols.keys()) #dividing by the overall number of protocols so we get the averaged tfidf
        sorted_collocations = [collocation for collocation, _ in sorted(collocations_by_tfidf.items(), key=lambda x: x[1], reverse=True)] #sort by tfidf value 
    else:
        raise ValueError(f'Error: Invalid type: {type}. type must be "frequency" or "tfidf"')
    #choosing the k most common collocations and returning them as sentences
    collocations_to_return = []
    for collocation in sorted_collocations[:min(k, len(sorted_collocations))]:
        joined_collocation = ' '.join(collocation)
        collocations_to_return.append(joined_collocation)
    return collocations_to_return

#A method that masks specific number of tokens in each sentence of a given sentences. 
#Input: List of sentences, rate of tokens to mask - x
#Output: List of sentences, in each x% of the tokens are masked
def mask_tokens_in_sentences(sentences, x):
    if not (0 < x < 100):
        raise ValueError("Error: The value of x must be in the range [0, 100]")
    masked_sentences = []
    for sentence in sentences:
        sentence_splitted = sentence.split()
        tokens_to_mask = (x/100)*len(sentence_splitted)
        tokens_to_mask = round(tokens_to_mask) if tokens_to_mask != 0.5 else 1 #computing the number of tokens to mask, rounding when the number is not integer, in addition, 0.5 rounded to 1 instead of 0. 
        indexes_to_replace = random.sample(range(len(sentence_splitted)), tokens_to_mask) #choosing "tokens_to_mask" random indexes to mask
        for idx in indexes_to_replace:
            sentence_splitted[idx] = '[*]'
        new_sentence = ' '.join(sentence_splitted)
        masked_sentences.append(new_sentence)
    return masked_sentences

def main():
    #if len(sys.argv) != 3:
    #    sys.exit(1)
    #file_path = sys.argv[1] 
    file_path = 'knesset_corpus.jsonl'
    #output_folder = sys.argv[2] 
    output_folder = 'output'
    #os.makedirs("example_dir", exist_ok=True)
    #reading the JSON file 
    try:
        corpus_df = pd.read_json(file_path, lines=True, encoding='utf-8')
    except FileNotFoundError:
        print(f'Error: The file {file_path} is not exist')
        return
    except ValueError:
        print(f'Error: The file {file_path} is not JSON valid file')
        return
    
    #Separating the committee and plenary data and exctracting the sentences related to each 
    try:
        committee_df = corpus_df[corpus_df['protocol_type'] == 'committee']
        plenary_df = corpus_df[corpus_df['protocol_type'] == 'plenary']
    except KeyError as e:
        print(f'Error: The column "protocol_type" is missing in the data: {e}')
        return 
    try:
        committee_sentences = committee_df['sentence_text'].tolist()
        plenary_sentences = plenary_df['sentence_text'].tolist()
    except KeyError as e:
        print(f'Error: The column "sentence_text" is missing in the data: {e}')
        return  

    #Creating instances of languages model
    try:
        committee_model = Trigram_LM(committee_sentences)
    except Exception as e:
        print(f'Error: Failed to create trigram model: comittee_model: {e}')
        return 
    try:
        plenary_model = Trigram_LM(plenary_sentences)
    except Exception as e:
        print(f'Error: Failed to create trigram model: plenary_model: {e}')
        return 
    
    #sections 2.2, 2.3, 2.4: printing the 10 most common collocations with threshold of 5, in each of the corpuses
    try:
        with open(os.path.join(output_folder, 'knesset_collocations.txt'), 'w', encoding='utf-8') as file:
            for n,n_str in [(2,'Two'),(3,'Three'),(4,'Four')]: #iterate over the longs of 2,3,4
                file.write(f'{n_str}-gram collocations:\n')
                for type_up_name, type in [('Frequency', 'frequency'), ('TF-IDF', 'tfidf')]:
                    file.write(f'{type_up_name}:\n')
                    for corpus_name, corpus_df in [('Committee corpus', committee_df), ('Plenary corpus', plenary_df)]:
                        file.write(f'{corpus_name}:\n')
                        collocations = get_k_n_t_collocations(10, n, 5, corpus_df, type)
                        for collocation in collocations:
                            file.write(f'{collocation}\n')
                        file.write('\n')
    except IOError as e:
        print(f'Error: Failed to write to "knesset_collocations.txt": {e}')

    #section 3.2: choosing 10 random messages, that are least 5 tokens long, from committee corpus and masking 10% of their tokens
    long_sentences = [sentence for sentence in committee_sentences if sentence.count(' ') >= 4]
    try:
        sentences_indexes = random.sample(range(len(long_sentences)), 10) #choosing 10 random indexes 
    except ValueError as e: #e.g., the case where number of sentences is less than 10 
        print(f'Error: Failed to sample sentences: {e}')
        sentences_indexes = []
    sentences_to_mask = [long_sentences[idx] for idx in sentences_indexes] #extracting the sentences in the previously chosen indexes
    try:
        with open(os.path.join(output_folder, 'original_sampled_sents.txt'), 'w', encoding='utf-8') as file:
            for sentence in sentences_to_mask:
                file.write(sentence+ '\n')
    except IOError as e:
        print(f'Error: Failed to write to "original_sampled_sents.txt": {e}')
    sentences_after_mask = mask_tokens_in_sentences(sentences_to_mask, 10)
    try:
        with open(os.path.join(output_folder, 'masked_sampled_sents.txt'), 'w', encoding='utf-8') as file:
            for sentence in sentences_after_mask:
                file.write(sentence+ '\n')
    except IOError as e:
        print(f'Error: Failed to write to "masked_sampled_sents.txt": {e}')

    #section 3.3: predicting the masked tokens using the committee model and computing the probability of the sentences
    sentences_after_mask_solve = []
    sentences_masked_indexes = [] #a list stores the masked tokens indexes for each sentences, this will be used in section 3.4
    try:
        with open(os.path.join(output_folder, 'sampled_sents_results.txt'), 'w', encoding='utf-8') as file:
            for i in range(10):
                file.write(f'original_sentence: {sentences_to_mask[i]}\n')
                file.write(f'masked_sentence: {sentences_after_mask[i]}\n')
                plenary_tokens = []
                sentence = sentences_after_mask[i].split()
                masked_indexes = [j for j,token in enumerate(sentence) if token == '[*]']
                sentences_masked_indexes.append(masked_indexes)
                for j in masked_indexes:
                    next_token, _ = plenary_model.generate_next_token(' '.join(sentence[:j]))
                    sentence[j] = next_token #replacing the [*] with the token predicted to be the masked one 
                    plenary_tokens.append(next_token)
                sentences_after_mask_solve.append(' '.join(sentence))
                file.write(f'plenary_sentence: {sentences_after_mask_solve[-1]}\n')
                plenary_tokens = ','.join(plenary_tokens)
                file.write(f'plenary_tokens: {plenary_tokens}\n')
                file.write(f'probability of plenary sentence in plenary corpus: {plenary_model.calculate_prob_of_sentence(sentences_after_mask_solve[i]):.2f}\n')
                file.write(f'probability of plenary sentence in committee corpus: {committee_model.calculate_prob_of_sentence(sentences_after_mask_solve[i]):.2f}\n')
    except IOError as e:
        print(f'Error: Failed to write to "sampled_sents_results.txt": {e}')

    #section 3.4: computing perplexity of the masked tokens in the sentences using the trigram perplexity formula
    try:
        with open(os.path.join(output_folder, 'perplexity_result.txt'), 'w', encoding='utf-8') as file:
            perplexity_average = 0
            for i in range(10):
                perplexity = 0
                sentence = ['s_0', 's_1'] + sentences_after_mask_solve[i].split()     
                for idx in sentences_masked_indexes[i]: #iterate over the masked tokens
                    prob = plenary_model.calculate_prob_of_token(sentence[idx:idx+3])
                    perplexity -= prob
                perplexity /= len(sentences_masked_indexes[i]) # dividing by the number of masked tokens 
                perplexity = 2 ** perplexity
                perplexity_average += perplexity / 10 
            file.write(f'{perplexity_average:.2f}\n')
    except IOError as e:
        print(f'Error: Failed to write to "perplexity_result.txt": {e}')

if __name__ == "__main__":
    main()
