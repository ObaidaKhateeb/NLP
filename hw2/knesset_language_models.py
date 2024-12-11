import json
import math 
import sys
import pandas as pd
import random

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
                if (sentence[i-2], sentence[i-1], sentence[i]) in trigrams_dict:
                    trigrams_dict[(sentence[i-2], sentence[i-1], sentence[i])] += 1
                else:
                    trigrams_dict[(sentence[i-2], sentence[i-1], sentence[i])] = 1
        return trigrams_dict
    def _bigrams_create(self, sentences):
        bigrams_dict = {}
        for sentence in sentences: 
            for i,token in enumerate(sentence[1:]):
                if (sentence[i-1], sentence[i]) in bigrams_dict:
                    bigrams_dict[(sentence[i-1], sentence[i])] += 1
                else:
                    bigrams_dict[(sentence[i-1], sentence[i])] = 1
        return bigrams_dict
    def _unigrams_create(self, sentences):
        unigrams_dict = {}
        for sentence in sentences:
            for i, token in enumerate(sentence):
                if token in unigrams_dict:
                    unigrams_dict[token] += 1
                else:
                    unigrams_dict[token] = 1
                if i not in [0,1]: 
                    self.tokens_count += 1
        self.unique_tokens_count = len(unigrams_dict.keys())
        if len(unigrams_dict.keys()):
            self.unique_tokens_count -= 2
        return unigrams_dict
    
    def calculate_prob_of_sentence(self, s):
        s = ['s_0', 's_1'] + s.split()
        total_prob = 0
        for i in range(2, len(s)):
            prob = 0
            denominator = numerator = 0
            #computing the probability of the trigram s[i-2] s[i-1] s[i]
            if (s[i-2], s[i-1]) in self.bigrams:
                denominator = self.bigrams[(s[i-2], s[i-1])] + self.unique_tokens_count
            else:
                denominator = self.unique_tokens_count
            if (s[i-2], s[i-1], s[i]) in self.trigrams:
                numerator = self.trigrams[(s[i-2], s[i-1], s[i])] + 1
            else:
                numerator = 1
            prob += lambda1 * numerator / denominator
            #computing the probability of the bigrams s[i-1] s[i]
            if s[i-1] in self.unigrams:
                denominator = self.unigrams[s[i-1]] + self.unique_tokens_count
            else:
                denominator = self.unique_tokens_count
            if (s[i-1], s[i]) in self.bigrams:
                numerator = self.bigrams[(s[i-1], s[i])] + 1
            else:
                numerator = 1
            prob += lambda2 * numerator / denominator
            #computing the probability of s[i]
            numerator = self.unigrams[s[i]] + 1 if s[i] in self.unigrams else 1
            prob += lambda3 * numerator / (self.tokens_count + self.unique_tokens_count)
            total_prob += math.log(prob)
        return total_prob
    def generate_next_token(self, s):
        s = s.split()
        next_token = None
        next_token_prob = float('-inf')
        for token in self.unigrams:
            s_new = ' '.join(s) + ' ' + token
            token_prob = self.calculate_prob_of_sentence(s_new)
            if token_prob > next_token_prob:
                next_token_prob = token_prob
                next_token = token
        return (next_token, math.exp(next_token_prob)) #exp of log of the probability will return the probability itself as required 

def get_k_n_t_collocations(k, n, t, corpus, type):
    collocations = {}
    protocols = {}
    #extracting all the collocations of long n
    for idx, row in corpus.iterrows():
        sentence_splitted = row['sentence_text'].split()
        protocol = row['protocol_number']
        if protocol not in protocols: 
            protocols[protocol] = {}
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
            if collocation in protocols[protocol]:
                protocols[protocol][collocation] += 1
            else:
                protocols[protocol][collocation] = 1
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
                tf = collocations[collocation][protocol] / sum(protocols[protocol].values()) #computing tf  
                idf = math.log(len(protocols.keys()) / len(collocations[collocation].keys())) #computing idf
                collocations_by_tfidf[collocation] += tf * idf 
            collocations_by_tfidf[collocation] /= len(protocols.keys()) #dividing by the overall number of protocols so we get the averaged tfidf
        sorted_collocations = [collocation for collocation, _ in sorted(collocations_by_tfidf.items(), key=lambda x: x[1], reverse=True)]
    #choosing the k most common collocations and returning them as sentences
    collocations_to_return = []
    for collocation in sorted_collocations[:min(k, len(sorted_collocations))]:
        joined_collocation = ' '.join(collocation)
        collocations_to_return.append(joined_collocation)
    return collocations_to_return

#A method that masks (x/100)
def mask_tokens_in_sentences(sentences, x):
    if not (0 < x < 100):
        raise ValueError("The value of x must be in the range [0, 100]")
    masked_sentences = []
    for sentence in sentences:
        sentence_splitted = sentence.split()
        tokens_to_mask = round((x/100)*len(sentence_splitted)) #computing the number of tokens to mask
        indexes_to_replace = random.sample(range(len(sentence_splitted)), tokens_to_mask) #choosing "tokens_to_mask" random indexes to mask
        for idx in indexes_to_replace:
            sentence_splitted[idx] = '[*]'
        new_sentence = ' '.join(sentence_splitted)
        masked_sentences.append(new_sentence)
    return masked_sentences

#global for check
lambda1 = 0.7
lambda2 = 0.29
lambda3 = 0.01

def main():
    global lambda1, lambda2, lambda3
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
    with open(file_path, 'r', encoding = 'utf-8') as file:
        lines_list =  [json.loads(line) for line in file]
    #Separating the committee and plenary data
    committee_sentences = [line['sentence_text'] for line in lines_list if line['protocol_type'] == 'committee']
    plenary_sentences = [line['sentence_text'] for line in lines_list if line['protocol_type'] == 'plenary']
    committee_model = Trigram_LM(committee_sentences)
    plenary_model = Trigram_LM(plenary_sentences)
    corpus_df = pd.DataFrame(lines_list)
    committee_df = corpus_df[corpus_df['protocol_type'] == 'committee']
    plenary_df = corpus_df[corpus_df['protocol_type'] == 'plenary']
    #sections 2.2, 2.3, 2.4: printing the 10 most common collocations with threshold of 5, in each of the corpuses
    # with open('knesset_collocations.txt', 'w', encoding = 'utf-8') as file:
    #     for n in [2,3,4]: #iterate over the longs of 2,3,4
    #         file.write(f'{n}-gram collocations:\n')
    #         for type_up_name, type in [('Frequency', 'frequency'), ('Tf-IDF', 'tfidf')]:
    #             file.write(f'{type_up_name}:\n')
    #             for corpus_name, corpus_df in [('Committee corpus', committee_df), ('Plenary corpus', plenary_df)]:
    #                 file.write(f'{corpus_name}:\n')
    #                 collocations = get_k_n_t_collocations(10, n, 5, corpus_df, type)
    #                 for collocation in collocations:
    #                     file.write(f'{collocation}\n')
    #                 file.write('\n')
    #         file.write('\n')
    #section 3.2: choosing 10 random messages from committee corpus and masking 10% of their tokens
    # sentences_indexes = random.sample(range(len(committee_df)), 10)
    # sentences_to_mask = [committee_df.iloc[idx]['sentence_text'] for idx in sentences_indexes]
    # with open('original_sampled_sents.txt', 'w', encoding = 'utf-8') as file:
    #     for sentence in sentences_to_mask:
    #         file.write(sentence+ '\n')
    # sentences_after_mask = mask_tokens_in_sentences(sentences_to_mask, 10)
    # with open('masked_sampled_sents.txt', 'w', encoding = 'utf-8') as file:
    #     for sentence in sentences_after_mask:
    #         file.write(sentence+ '\n')
    #section 3.3: predicting the masked tokens using the committee model and computing the probability of the sentences
    # sentences_after_mask_solve = sentences_after_mask[:]
    # subsentences_after_mask_solve = []
    # with open('sampled_sents_results.txt', 'w', encoding = 'utf-8') as file:
    #     for i in range(10):
    #         file.write(f'original_sentence: {sentences_to_mask[i]}\n')
    #         file.write(f'masked_sentence: {sentences_after_mask[i]}\n')
    #         plenary_tokens = []
    #         subsentences = [] #for section 3.4
    #         masked_idx = sentences_after_mask_solve[i].find('[*]')
    #         while(masked_idx != -1):
    #             next_token, _ = plenary_model.generate_next_token(sentences_after_mask_solve[i][:masked_idx])
    #             sentences_after_mask_solve[i] = sentences_after_mask_solve[i][:masked_idx] + next_token + sentences_after_mask_solve[i][masked_idx + 3:]
    #             plenary_tokens.append(next_token)
    #             subsentences.append(sentences_after_mask_solve[i][:masked_idx] + next_token)
    #             masked_idx = sentences_after_mask_solve[i].find('[*]')
    #         subsentences_after_mask_solve.append(subsentences)
    #         file.write(f'plenary_sentence: {sentences_after_mask_solve[i]}\n')
    #         plenary_tokens = ','.join(plenary_tokens)
    #         file.write(f'plenary_tokens: {plenary_tokens}\n')
    #         file.write(f'probability of plenary sentence in plenary corpus: {plenary_model.calculate_prob_of_sentence(sentences_after_mask_solve[i]):.2f}\n')
    #         file.write(f'probability of plenary sentence in committee corpus: {committee_model.calculate_prob_of_sentence(sentences_after_mask_solve[i]):.2f}\n')
    #section 3.4: computing perplexity of the masked tokens in the sentences using the trigram perplexity formula
    # with open('perplexity_result.txt', 'w', encoding = 'utf-8') as file:
    #     for i in range(10):
    #         perplexity = 1
    #         if not subsentences_after_mask_solve[i]: #the case when no masked tokens, due to the sentence being short and 10% of it is rounded to 0
    #             file.write('None\n')
    #         else:
    #             for subsentence in subsentences_after_mask_solve[i]: #iterate over the masked tokens
    #                 perplexity *= (1 / plenary_model.calculate_prob_of_sentence(subsentence)) # perplexity *= 1/P(masked token | subsentence until masked token)
    #             perplexity = perplexity ** (1/ len(subsentences_after_mask_solve[i])) #perplexity = perplexity^(1/n)
    #             file.write(f'{perplexity:.2f}\n')
        
    #checks 
    sentence_prob = committee_model.calculate_prob_of_sentence('אחמד טיבי')
    print(sentence_prob)
    next_word = committee_model.generate_next_token('אחמד')
    print(next_word)
    relevant_sentences = get_k_n_t_collocations(5, 5, 5, committee_df, 'frequency')
    for i in range(len(relevant_sentences)):
        print (relevant_sentences[i])
    #checking masking: 
    original_sentences_indexes = random.sample(range(len(committee_df)), 100)
    original_sentences = [committee_df.iloc[idx]['sentence_text'] for idx in original_sentences_indexes]
    sentences_after_mask = mask_tokens_in_sentences(original_sentences, 10)
    for idx1 in range(20, -1, -1):
        lambda1 = 0.05* idx1
        for idx2 in [20-idx1, 20-idx1 - 0.2, 20-idx1 - 0.02, 20-idx1 - 0.002, 20-idx1 - 0.0002, 20-idx1 - 0.00002]:
            if idx2 < 0.000001:
                continue
            lambda2 = 0.05*idx2
            lambda3 = 1 - lambda1 - lambda2
            print(f"lambda1 = {lambda1:.2f}, lambda2 = {lambda2}, lambda3 = {lambda3}")
            hits = 0
            misses = 0
            comma_hits = 0
            comma_misses = 0
            # original_sentences = ['הוא אמר : זה לשיקולך', 
            #                     'נציב שירות המדינה – לא מנכ"ל משרד , לא מנהל בית ספר , לא מנהל בית חולים – נציב שירות המדינה , ואין הסמכות כלפי מטה בזה – זכאי לקבל דיווח על פתיחה בחקירה נגד עובד מדינה או עובד שכפוף לחוק המשמעת בשביל לבחון את סמכותו על - פי דין להשעות את אותו עובד .', 
            #                     '"אתם יודעים מה זה להחזיק ילדים בגנים פרטיים ?', 
            #                     'היא מנהלת את פרויקט שיקום עוטף עזה במאות מיליונים , שפועל למיטב הכרתי – אני אולי לא אובייקטיבי – אבל הפרק הזה פועל היטב .', 
            #                     'על פי חוק התכנון והבנייה .', 
            #                     'האמת היא , שזה לא חייב להוריד את המחיר .', 
            #                     'זה מזכיר לי את הפעם שהפגנתי מול משרד הביטחון בתל - אביב בתקופת הסכמי אוסלו , והמשטרה הדפה אותי .', 
            #                     'אני פותח את הדיון במליאת הכנסת ביום המיוחד הזה , שיוחד לשפה הערבית , וזה מאורע שלא היה כמותו .', 
            #                     'התוכנית הוכנה בשיתוף פעולה ובתיאום בין כל הגורמים השותפים למאבקנו , המאבק בתאונות הדרכים .',
            #                     'אני רוצה לקבל חוות דעת .'
            #                     'אני קורא אותך לסדר פעם ראשונה .'
            #                     'הנתונים הם לא נתונים שלנו , הם נתונים של המוסד לביטוח לאומי , שאומרים שזה בסך הכול 6.', 
            #                     'אני מתכבד לפתוח את ישיבת הכנסת .', 
            #                     'חבר הכנסת אחמד טיבי , בבקשה .'
            #                     'האם ממשלה זו לא יוצרת את בעיית האבטלה במגזר הערבי ואת בעיית העוני ואת בעיית הכפרים הלא - מוכרים ?', 
            #                     'מה שאמר עמיר פרץ זה נכון .', 
            #                     'אדוני היושב - ראש , אני מודה לך על שאפשרת לי .']
            # masked_sentences = ['הוא אמר [*] זה לשיקולך', 'ציב שירות המדינה – לא מנכ"ל משרד , לא מנהל בית [*] , לא מנהל בית [*] – נציב שירות [*] , ואין הסמכות כלפי מטה בזה – זכאי לקבל דיווח על פתיחה בחקירה נגד עובד מדינה או עובד שכפוף לחוק המשמעת בשביל לבחון את סמכותו על - פי דין להשעות את אותו עובד .', '"אתם יודעים [*] זה להחזיק ילדים בגנים פרטיים ?', 'היא מנהלת את פרויקט שיקום עוטף [*] במאות [*] , שפועל למיטב הכרתי – אני אולי לא אובייקטיבי – אבל הפרק הזה פועל [*] .', 'על פי [*] התכנון והבנייה .', 'האמת היא , שזה לא חייב להוריד את [*] .', 'זה מזכיר לי את הפעם שהפגנתי מול משרד הביטחון בתל - [*] בתקופת הסכמי [*] , והמשטרה הדפה אותי .', 'אני פותח את הדיון במליאת [*] ביום המיוחד הזה , שיוחד לשפה הערבית , וזה מאורע שלא היה כמותו .', 'התוכנית הוכנה בשיתוף [*] ובתיאום בין כל הגורמים השותפים למאבקנו , המאבק בתאונות [*] .', 'אני רוצה לקבל חוות [*] .', 'אני קורא אותך לסדר [*] ראשונה .', 'הנתונים הם לא נתונים שלנו , הם נתונים של המוסד לביטוח [*] , שאומרים שזה בסך הכול 6.', 'אני מתכבד לפתוח את [*] הכנסת .', 'חבר הכנסת אחמד [*] , בבקשה .', 'האם ממשלה זו לא יוצרת את בעיית האבטלה במגזר [*] ואת בעיית העוני ואת בעיית הכפרים הלא [*] מוכרים ?', 'מה שאמר עמיר [*] זה נכון .', 'אדוני [*] - ראש , אני מודה [*] על שאפשרת לי .']
            # masked_tokens = [[':'], ['ספר', 'חולים', 'המדינה'], ['מה'], ['עזה', 'מיליונים', 'היטב'], ['חוק'], ['המחיר'], ['אביב', 'אוסלו'], ['הכנסת'], ['פעולה', 'הדרכים'], ['דעת'], ['פעם'], ['לאומי'], ['ישיבת'], ['טיבי'], ['הערבי', '-'], ['פרץ'], ['היושב', 'לך']]
            for i,sentence in enumerate(sentences_after_mask):
                #token_idx = 0
                dup_sentence = sentence[:]
                masked_idx = dup_sentence.find('[*]')
                while(masked_idx != -1):
                    next_token, probability = plenary_model.generate_next_token(sentence[:masked_idx])
                    if sentence[:masked_idx] + next_token == original_sentences[i][:masked_idx + len(next_token)]:
                        hits += 1
                        print(f"TRUE token : {next_token}, probability = {probability}")
                        dup_sentence = dup_sentence[:masked_idx] + next_token + dup_sentence[masked_idx + 3:]
                        if next_token == ',':
                            comma_hits += 1
                    else:
                        misses += 1
                        #print(f"FALSE token : {next_token}, probability = {probability}")
                        dup_sentence = original_sentences[i]
                        if next_token == ',':
                            comma_misses += 1
                    #dup_sentence = dup_sentence[masked_idx + 3:]
                    masked_idx = dup_sentence.find('[*]')
                    #token_idx += 1
            print(f"hits = {hits}, misses = {misses}, success rate = {hits/(hits+misses)}\n")
            print(f"comma hits = {comma_hits}, comma misses = {comma_misses}\n")
            
if __name__ == "__main__":
    main()