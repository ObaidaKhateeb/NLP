first_knesset_numbers = {13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 23: 0, 25: 0}
second_knesset_numbers = {13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 23: 0, 25: 0}
others_knesset_numbers = {13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 23: 0, 25: 0}
first_protocol_types = {'committee': 0 , 'plenary': 0}
second_protocol_types = {'committee': 0 , 'plenary': 0}
others_protocol_types = {'committee': 0 , 'plenary': 0}
first_unigrams = {}
second_unigrams = {}
others_unigrams = {}
first_bigrams = {}
second_bigrams = {}
others_bigrams = {}
first_trigrams = {}
second_trigrams = {}
others_trigrams = {}
first_4grams = {}
second_4grams = {}
others_4grams = {}
first_5grams = {}
second_5grams = {}
others_5grams = {}
for sentence_collection in [
    (first_sentences, first_unigrams, first_bigrams, first_trigrams, first_4grams, first_5grams),
    (second_sentences, second_unigrams, second_bigrams, second_trigrams, second_4grams, second_5grams),
    (others_sentences, others_unigrams, others_bigrams, others_trigrams, others_4grams, others_5grams)
]:
    for sentence in sentence_collection[0]:
        # Update protocol type counts
        if sentence.protocol_type == 'committee':
            first_protocol_types['committee'] += 1
        else:
            first_protocol_types['plenary'] += 1
        
        # Update knesset numbers
        if sentence.keneset in first_knesset_numbers:
            first_knesset_numbers[sentence.keneset] += 1
        
        sentence_splitted = sentence.text.split()

        # Update unigrams
        for i in range(1, len(sentence_splitted) + 1):
            word = sentence_splitted[i-1]
            if word in sentence_collection[1]:
                sentence_collection[1][word] += 1
            else:
                sentence_collection[1][word] = 1

        # Update bigrams
        for i in range(2, len(sentence_splitted) + 1):
            bigram = sentence_splitted[i-2] + ' ' + sentence_splitted[i-1]
            if bigram in sentence_collection[2]:
                sentence_collection[2][bigram] += 1
            else:
                sentence_collection[2][bigram] = 1

        # Update trigrams
        for i in range(3, len(sentence_splitted) + 1):
            trigram = sentence_splitted[i-3] + ' ' + sentence_splitted[i-2] + ' ' + sentence_splitted[i-1]
            if trigram in sentence_collection[3]:
                sentence_collection[3][trigram] += 1
            else:
                sentence_collection[3][trigram] = 1

        # Update 4-grams
        for i in range(4, len(sentence_splitted) + 1):
            fourgram = (
                sentence_splitted[i-4] + ' ' + sentence_splitted[i-3] + ' ' + sentence_splitted[i-2] + ' ' + sentence_splitted[i-1]
            )
            if fourgram in sentence_collection[4]:
                sentence_collection[4][fourgram] += 1
            else:
                sentence_collection[4][fourgram] = 1

        # Update 5-grams
        for i in range(5, len(sentence_splitted) + 1):
            fivegram = (
                sentence_splitted[i-5] + ' ' + sentence_splitted[i-4] + ' ' + sentence_splitted[i-3] + ' ' + sentence_splitted[i-2] + ' ' + sentence_splitted[i-1]
            )
            if fivegram in sentence_collection[5]:
                sentence_collection[5][fivegram] += 1
            else:
                sentence_collection[5][fivegram] = 1


#Protocol type feature check: 
print('Protocol types count of each class:')
print(f'first: {first_protocol_types}')
print(f'second: {second_protocol_types}')
print(f'others: {others_protocol_types}')
#Knesset number feature check: 
print('Knesset numbers count of each class:')
for knesset in [13, 14, 15, 16, 17, 18, 19, 20, 23, 25]:
    print(f'knesset {knesset}:')
    print(f'first: {first_knesset_numbers[knesset]}, second: {second_knesset_numbers[knesset]}, others: {others_knesset_numbers[knesset]}')
#N-grams feature check:
print('Unigrams count of each class:')
sorted_first_unigrams = sorted(first_unigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_second_unigrams = sorted(second_unigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_others_unigrams = sorted(others_unigrams.items(), key=lambda x: x[1], reverse=True)[:10]
print(f'first: {sorted_first_unigrams}')
print(f'second: {sorted_second_unigrams}')
print(f'others: {sorted_others_unigrams}')
print('Bigrams count of each class:')
sorted_first_bigrams = sorted(first_bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_second_bigrams = sorted(second_bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_others_bigrams = sorted(others_bigrams.items(), key=lambda x: x[1], reverse=True)[:10]
print(f'first: {sorted_first_bigrams}')
print(f'second: {sorted_second_bigrams}')
print(f'others: {sorted_others_bigrams}')
print('Trigrams count of each class:')
sorted_first_trigrams = sorted(first_trigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_second_trigrams = sorted(second_trigrams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_others_trigrams = sorted(others_trigrams.items(), key=lambda x: x[1], reverse=True)[:10]
print(f'first: {sorted_first_trigrams}')
print(f'second: {sorted_second_trigrams}')
print(f'others: {sorted_others_trigrams}')
print('4-grams count of each class:')
sorted_first_4grams = sorted(first_4grams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_second_4grams = sorted(second_4grams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_others_4grams = sorted(others_4grams.items(), key=lambda x: x[1], reverse=True)[:10]
print(f'first: {sorted_first_4grams}')
print(f'second: {sorted_second_4grams}')
print(f'others: {sorted_others_4grams}')
print('5-grams count of each class:')
sorted_first_5grams = sorted(first_5grams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_second_5grams = sorted(second_5grams.items(), key=lambda x: x[1], reverse=True)[:10]
sorted_others_5grams = sorted(others_5grams.items(), key=lambda x: x[1], reverse=True)[:10]
print(f'first: {sorted_first_5grams}')
print(f'second: {sorted_second_5grams}')
print(f'others: {sorted_others_5grams}')


for word in ['אני', 'הכנסת', 'חבר הכנסת', 'חברי הכנסת', 'לחבר הכנסת', 'הצעת חוק', 'הצעת החוק', 'רבותי חברי', 'כהצעת הוועדה', 'הצעת הוועדה', 'ההסתייגות של', 'אדוני היור', 'אדוני היו”ר', 'אדוני היושב', 'היושב ראש', 'אדוני היושב ראש', 'אדוני היושב - ראש', 'יושב - ראש', 'היושב - ראש', 'רבותי חברי הכנסת', 'בבקשה', 'תודה', 'תודה רבה']:
    print(f'{word}:')
    if word in first_unigrams:
        print(f'first: {first_unigrams[word]}')
    if word in second_unigrams:
        print(f'second: {second_unigrams[word]}')
    if word in others_unigrams:
        print(f'others: {others_unigrams[word]}')
    if word in first_bigrams:
        print(f'first: {first_bigrams[word]}')
    if word in second_bigrams:
        print(f'second: {second_bigrams[word]}')
    if word in others_bigrams:
        print(f'others: {others_bigrams[word]}')
    if word in first_trigrams:
        print(f'first: {first_trigrams[word]}')
    if word in second_trigrams:
        print(f'second: {second_trigrams[word]}')
    if word in others_trigrams:
        print(f'others: {others_trigrams[word]}')
    if word in first_4grams:
        print(f'first: {first_4grams[word]}')
    if word in second_4grams:
        print(f'second: {second_4grams[word]}')
    if word in others_4grams:
        print(f'others: {others_4grams[word]}')
    if word in first_5grams:
        print(f'first: {first_5grams[word]}')
    if word in second_5grams:
        print(f'second: {second_5grams[word]}')
    if word in others_5grams:
        print(f'others: {others_5grams[word]}')