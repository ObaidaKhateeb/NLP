import os
from docx import Document
import json
import sys
import re 

class Sentence:
    def __init__(self, protocol_name, keneset, protocol_type, protocol_no, speaker, text):
        self.protocol_name = protocol_name
        self.keneset = keneset
        self.protocol_type = protocol_type
        self.protocol_no = protocol_no
        self.speaker = speaker
        self.text = text
    
class Protocol:
    def __init__(self, name, keneset, type, number):
        self.name = name
        self.keneset = keneset
        self.type = type
        self.number = number
        self.sentences = {}
    def add_sentence(self, speaker, sentence):
        if speaker in self.sentences:
            self.sentences[speaker].append(sentence)
        else:
            self.sentences[speaker] = [sentence]
    def check(self): #function for tests .. will be deleted later 
        print(self.name)
        print(len(self.sentences.keys()))
        sorted_keys = sorted(self.sentences.keys(), key=lambda name: name.split()[-1])
        for key in sorted_keys:
            print(key)

ministries = {'התשתיות הלאומיות', 'התשתיות', 'המשטרה', 'לביטחון פנים', 'לביטחון הפנים', 
              'לאיכות הסביבה', 'להגנת הסביבה', 'החינוך, התרבות והספורט', 'החינוך והתרבות', 
              'החינוך', 'התחבורה והבטיחות בדרכים', 'התחבורה', 'האוצר', 'הכלכלה והתכנון', 
              'הכלכלה', 'המשפטים', 'הבריאות', 'החקלאות ופיתוח הכפר', 'החקלאות', 'החוץ', 
              'הבינוי והשיכון', 'העבודה, הרווחה והשירותים החברתיים', 'העבודה והרווחה', 
              'העבודה', 'הרווחה', 'התעשייה והמסחר', 'התעשייה, המסחר והתעסוקה', 'התיירות', 
              'המדע והטכנולוגיה', 'הפנים', 'המדע, התרבות והספורט', 'התרבות והספורט', 
              'האנרגיה והמים', 'לענייני דתות', 'במשרד ראש הממשלה', 
              'לנושאים אסטרטגיים ולענייני מודיעין', 'לקליטת העלייה', 'לאזרחים ותיקים', 
              'במשרד', 'הביטחון', 'המודיעין', 'התקשורת', 'התשתיות הלאומיות, האנרגיה והמים'}
titles_and_symbols = ['<', '>', 'היו"ר', 'היו”ר', 'יו"ר הכנסת', 'יו”ר הכנסת', 
                      'יו"ר ועדת הכנסת', 'יו”ר ועדת הכנסת', 'מ"מ', 'מ”מ', 'סגן', 'סגנית', 
                      'מזכיר הכנסת', 'מזכירת הכנסת', 'תשובת', 'המשנה לראש הממשלה', 'ראש הממשלה', 
                      'עו"ד', 'עו”ד', 'ד"ר', 'ד”ר', "פרופ'", 'נצ"מ', 'ניצב']
invalid_names = {'2', 'ברצוני', 'כרצוני', 'רצוני', 'אני', 'אחרי', 'הצעת', 'המועצה', 'ביום', 
                 'בפסקה', 'קריאה', 'קריאות', 'האפשרות', 'קוראת', 'קורא', 'הצעת', 'מסקנות', 
                 'להלן'}
name_duplicates = {"םשה גפני" : "משה גפני", "1 סאלח טריף" : "סאלח טריף", 
                   "אופיר פינס" : "אופיר פינס פז", "י יוסף ג'בארין" : "יוסף ג'בארין", 
                   "אורה עשאהל זילברשטיין" : "אורה עשהאל זילברשטיין", "שלי יחימוביץ'" : "שלי יחימוביץ", 
                   'שמרית שקד גיטלין' : 'שמרית גיטלין שקד', "חוסנייה ג'בארה" : "חוסניה ג'בארה", 
                   "חנא סוייד" : "חנא סוויד", "מיכאל נולדמן" : "מיכאל נודלמן", 
                   "מל פולישוק בלוך" : "מלי פולישוק בלוך", "מריה רבינוביץ'" : "מריה רבינוביץ",
                   "נאווה רצון" : "נאוה רצון", "נדיה חילו" : "נאדיה חילו", 
                   "רובי ריבלין:" : "ראובן ריבלין", "רובי ריבלין": "ראובן ריבלין",
                   "שימרית גיטלין שקד" : "שמרית גיטלין שקד", "אייל בן ראובן" : "איל בן ראובן",
                   "מרים פרנקל שור" : "מירי פרנקל שור", "בצלאל סמוטריץ'" : "בצלאל סמוטריץ", 
                   "בועז בועז מקלר" : "בועז מקלר", "יולי יואל אדלשטיין" : "יולי אדלשטיין", 
                   "יוסף ביילין" : "יוסי ביילין", "רונית אנדרלט" : "רונית אנדוולט", 
                   "אברהם בייגה שוחט" : "אברהם שוחט"}
patterns = [r'(\b\w+\b)[,\s]*(\b\w+\b)[,\s]*(\b\w+\b)[,\s]*(\b\w+\b)', 
            r'(\b\w+\b)[,\s]*(\b\w+\b)[,\s]*(\b\w+\b)', r'(\b\w+\b)[,\s]*(\b\w+\b)', r'(\b\w+\b)']
digits_dict = {'אחת': 1, 'שתיים': 2, 'שתים' : 2, 'שלוש': 3, 'ארבע': 4, 'חמש': 5, 'חמיש': 5, 
               'שש': 6, 'שיש': 6, 'שבע': 7, 'שמונה': 8, 'תשע': 9, 'עשר': 10, 'עשרים': 20, 
               'שמונים': 80, 'מאה': 100, 'מאתיים': 200, 'אלף': 1000}
speakers_dict = {}

#Helper method for 'extract_metada_from_content'. It checks if the given string represent an integer number, and if yes, it return its integer value
#Input: String of number which can be numeric or as hebrew word
#Output: Integer equivalent of the number or -1 in case it's not identified as a number
def convertToInt(word):
    # case 1: string in numeric form
    if word.isdigit():
        return int(word)
    elif word[:-1].isdigit(): #to deal with the cases where the number followed by '<' directly
        return int(word[:-1])
    #case 3: string in string form 
    else:
        word_splitted = word.split('-')
        word_splitted = [word[1:] if word[0] in ['ו', 'ה']else word for word in word_splitted]
        for i in range(len(word_splitted)):
            #option 1: digit from the digits dictionary 
            if word_splitted[i] in digits_dict:
                word_splitted[i] = digits_dict[word_splitted[i]]
            #option 2: digit in plural form
            elif word_splitted[i][-2:] == 'ים' and word_splitted[i][:-2] in digits_dict:
                word_splitted[i] = digits_dict[word_splitted[i][:-2]] * 10
            #option 3: the word 'מאות'
            elif word_splitted[i] == 'מאות' and i > 0:
                word_splitted[i-1] *= 100
                word_splitted[i] = 0
            #option 4: the word 'עשרה'
            elif word_splitted[i] == 'עשרה' and i > 0:
                word_splitted[i-1] += 10
                word_splitted[i] = 0
            #option 5: the string didn't identify as a word
            else:
                return -1
        return sum(word_splitted)

#Input: string which could represent someone's name 
#Output: the name without the additions and titles 
def speakerClean(speaker):
    #remove the party name
    open_brac = speaker.find('(')
    close_brac = speaker.find(')')
    if open_brac != -1 and close_brac != -1:
        speaker = speaker[:open_brac] + speaker[close_brac+1:]
    #remove the tag
    open_an_brac = speaker.find('<<')
    close_an_brac = speaker.find('>>')
    while(open_an_brac != -1 and close_an_brac != -1):
        speaker = speaker[:open_an_brac] + speaker[close_an_brac+1:]
        open_an_brac = speaker.find('<<')
        close_an_brac = speaker.find('>>')
    #remove the title and other symbols
    for element in titles_and_symbols:
        speaker = speaker.replace(element, '')
    #replace the '-' by a space
    speaker = speaker.replace('-', ' ')
    #replace multi spaces by one space
    speaker = speaker.replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')
    #remove minister titles, these titles are special case because they can be followed by the ministry name 
    if 'השר ' in speaker or 'שרת ' in speaker or 'השרה ' in speaker:     
        speaker = speaker.replace('השרה ','').replace('השר ','').replace('שרת ','').replace('שר ','')
        for pattern in patterns:
            ministry_name = re.search(pattern, speaker)
            if ministry_name and ministry_name.group(0) in ministries:
                speaker = speaker[ministry_name.end():].strip()
                break
    speaker = speaker.strip()
    #remove minister title only if it appears in the beginning, this to avoid ditortion of first names ends with this suffix like 'Asher'
    if 'שר ' in speaker and speaker.find('שר ') == 0: 
        speaker = speaker.replace('שר ', '')
        for pattern in patterns:
            ministry_name = re.search(pattern, speaker)
            if ministry_name and ministry_name.group(0) in ministries:
                speaker = speaker[ministry_name.end():].strip()
                break
        speaker = speaker.strip()
    return speaker

#Input: file/protocol name
# Output: The keneset number to where the protocol relates, protocol type       
def extract_metada_from_name(file_name):
    file_name_splitted = file_name.split('_')
    keneset_no = int(file_name_splitted[0])
    protocol_type = 'committee' if file_name_splitted[1][-1] == 'v' else 'plenary' if file_name_splitted[1][-1] == 'm' else None
    return keneset_no, protocol_type

#Input: .docx file of a protocol
#Output: protocol number if found, -1 otherwise. 
def extract_metada_from_content(file_content):
    for paragraph in file_content.paragraphs:
        paragraph_stripped = paragraph.text.strip()
        #iterates over the occurrences of the two string until it found a one that's followed by a number
        for word in ["הישיבה", "פרוטוקול מס'"]:
            while True:
                word_idx = paragraph_stripped.find(word)
                if word_idx != -1:
                    paragraph_stripped = paragraph_stripped[word_idx+len(word):] #slicing the paragraph to the part after the occurence
                    paragraph_words = paragraph_stripped.split()
                    if paragraph_words:
                        string_to_int = convertToInt(paragraph_words[0]) #if the string followed by a number it should be the first in the new sliced paragraph
                    else:
                        continue
                    if string_to_int != -1: #if a number identified after the string
                        return string_to_int 
                    else:
                        continue 
                else:
                    break
    return -1

#helper method for for extract_relevant_text, it identifies the start of the relevant text
#Input: .docx file of a protocol
#Output: index of the first relevant paragraph
def find_starting_relevant(file_content): 
    paragraph_idx = 0
    for paragraph_idx in range(len(file_content.paragraphs)):
        paragraph_txt = file_content.paragraphs[paragraph_idx].text.strip()
        if (paragraph_txt.startswith('היו"ר') or paragraph_txt.startswith('היו”ר') or paragraph_txt.startswith('יו"ר הכנסת') or paragraph_txt.startswith('יו”ר הכנסת') or paragraph_txt.startswith('מ"מ היו"ר') or paragraph_txt.startswith('מ”מ היו”ר')) and paragraph_txt.endswith(':'):
            return paragraph_idx
        elif paragraph_txt.startswith('<< יור >>') and paragraph_txt.endswith('<< יור >>'):
            return paragraph_idx
        elif paragraph_txt.startswith('<היו"ר') and paragraph_txt.endswith(':>'):
            return paragraph_idx
    return paragraph_idx

#helper method for for extract_relevant_text, it identifies the last relevant paragraph
#Input: .docx file of a protocol
#Output: index of the last relevant paragraph
def find_last_relevant(file_content): 
    idx = len(file_content.paragraphs) - 1
    curr_paragraph = file_content.paragraphs[idx].text
    while (not len(curr_paragraph) or ('הישיבה ננעלה' not in curr_paragraph and 'הטקס ננעל' not in curr_paragraph)) and idx > 0:
        idx -= 1
        curr_paragraph = file_content.paragraphs[idx].text
    if idx == 0: #when it fails to find a message states debate ending explicitly, it assumes that the last paragraph in the protocol is the last relevant
        return idx - 1 
    else: 
        return idx

#helper method for sentence_handle which checks validity of a sentence
def sentence_validity(sentence):
    if all(not('א' <= letter <= 'ת') for letter in sentence):
        return False
    if any('a' <= letter <= 'z' or 'A' <= letter <= 'Z' for letter in sentence):
        return False
    if '- -' in sentence or '– –' in sentence or '...' in sentence:
        return False
    return True

def sentence_tokenize(sentence):
    marks = {',', ';', ':', '(', ')', ' '}
    tokenized_sentence = []
    word = ''
    for i,letter in enumerate(sentence):
        if letter in marks:
            if word:
                tokenized_sentence.append(word)
                word = ''
            if letter != ' ':
                tokenized_sentence.append(letter)
        elif letter == '"' and (i == 0 or sentence[i-1] == ' ' or i == len(sentence)-1 or sentence[i+1] == ' '):
            if word:
                tokenized_sentence.append(word)
                tokenized_sentence.append(letter)
                word = ''
        else:
            word += letter
    if word:
        tokenized_sentence.append(word)
    if len(tokenized_sentence) >= 4:
        tokenized_sentence = ' '.join(tokenized_sentence)
        return tokenized_sentence
    else:
        return None

#helper method for for extract_relevant_text that creates a sentence object and add it to the relevant protocol 
def sentence_handle(protocol, curr_speaker, paragraph_txt):
    curr_sentence = ''
    for i,letter in enumerate(paragraph_txt): 
        if letter in ['?', '!']:
            if sentence_validity(curr_sentence):
                curr_sentence = curr_sentence.replace('\"', '"') #to fix the problem where many times '"' appears as '\"' 
                curr_sentence = sentence_tokenize(curr_sentence)
                if curr_sentence:
                    sentence = Sentence(protocol.name, protocol.keneset, protocol.type, protocol.number, curr_speaker, curr_sentence)
                    protocol.add_sentence(curr_speaker, sentence)
            curr_sentence = ''
        elif letter in ['.']:
            #case 1: '.' is a decimal point, not considered as an end of a sentence
            if len(curr_sentence) > 0 and i != len(paragraph_txt) - 1 and curr_sentence[-1].isdigit() and paragraph_txt[i+1].isdigit():
               curr_sentence += letter
            #case 2: '.' is a period, precedeed by a number, not considered as an end of a sentence
            elif len(curr_sentence) > 2 and curr_sentence[-1].isdigit() and curr_sentence[-2] == ' ' and curr_sentence[-3] in [',' , ':']:
                curr_sentence += letter
            elif len(curr_sentence) == 1 and curr_sentence[-1].isdigit():
                curr_sentence += letter
            #case 3: '.' is a period, precedeed by a letter, not considered as an end of a sentence
            elif len(curr_sentence) > 1 and  'א' <= curr_sentence[-1] <= 'י' and curr_sentence[-2] == ' ':
                curr_sentence += letter
            #case 4: '.' is an end of a sentence
            else:
                if sentence_validity(curr_sentence):
                    curr_sentence = curr_sentence.replace('\"', '"')
                    curr_sentence = sentence_tokenize(curr_sentence)
                    if curr_sentence:
                        sentence = Sentence(protocol.name, protocol.keneset, protocol.type, protocol.number, curr_speaker, curr_sentence)
                        protocol.add_sentence(curr_speaker, sentence)
                curr_sentence = ''
        elif not (letter == ' ' and curr_sentence == ''): #to avoid a sentence starting with spaces 
            curr_sentence += letter

#extracts the relevant sentences and arranges them according to protocol and speaker
def extract_relevant_text(file_content, protocol):
    #find the first and last paragraphs of the block where relevant texts 
    first_idx = find_starting_relevant(file_content)
    last_idx = find_last_relevant(file_content)
    curr_speaker = None
    for i,paragraph in enumerate(file_content.paragraphs[first_idx:last_idx]):
        #type 1 of sentences to exclude: those related to 'vote'. If it's a vote sentence, consider the text irrelevant until there's a speaker
        if 'הצבעה' in paragraph.text or 'ההצבעה' in paragraph.text or 'הצבעת' in paragraph.text:
            j = i+1
            while not file_content.paragraphs[first_idx+j].text:
                j += 1
            if 'בעד' in file_content.paragraphs[first_idx + j].text and 'נגד' in file_content.paragraphs[first_idx + j+1].text:
                curr_speaker = None
        #type 2 of sentences to exclude: those related to 'debate pause'. If it so, consider the paragraph irrelevant until there's a speaker
        if 'הישיבה נפסקה' in paragraph.text:
            curr_speaker = None
        paragraph_txt = paragraph.text.strip()
        #meeting one of the first two if's making the paragraph as potential speaker's name
        if paragraph_txt.endswith(':') or paragraph_txt.endswith(':>'):
            paragraph_txt_cleaned = speakerClean(paragraph_txt)[:-1].strip()
            first_string = re.search(r'\b\w+\b', paragraph_txt_cleaned)
            if len(paragraph_txt_cleaned.split()) < 6 and first_string and first_string.group(0) not in invalid_names:
                curr_speaker = paragraph_txt_cleaned if paragraph_txt_cleaned not in name_duplicates else name_duplicates[paragraph_txt_cleaned]
                curr_speaker_last = curr_speaker.split()[-1]
                if curr_speaker_last in speakers_dict:
                    speakers_dict[curr_speaker_last].append(curr_speaker)
                else:
                    speakers_dict[curr_speaker_last] = [curr_speaker]
            elif curr_speaker: #deal with it as speech
                sentence_handle(protocol, curr_speaker, paragraph_txt)
        elif ':' in paragraph_txt:
            pot_curr_speaker = speakerClean(paragraph_txt)
            if pot_curr_speaker.endswith(':'):
                paragraph_txt_cleaned = pot_curr_speaker[:-1].strip()
                first_string = re.search(r'\b\w+\b', paragraph_txt_cleaned)
                if len(paragraph_txt_cleaned.split()) < 6 and first_string and first_string.group(0) not in invalid_names:
                    curr_speaker = paragraph_txt_cleaned if paragraph_txt_cleaned not in name_duplicates else name_duplicates[paragraph_txt_cleaned]
                    curr_speaker_last = curr_speaker.split()[-1]
                    if curr_speaker_last in speakers_dict:
                        speakers_dict[curr_speaker_last].append(curr_speaker)
                    else:
                        speakers_dict[curr_speaker_last] = [curr_speaker]
                elif curr_speaker: #deal with it as speech
                    sentence_handle(protocol, curr_speaker, paragraph_txt)
            elif curr_speaker: #else deal with it as a speech 
                sentence_handle(protocol, curr_speaker, paragraph_txt)
        elif curr_speaker:
            sentence_handle(protocol, curr_speaker, paragraph_txt)

def speaker_full_name(speaker1):
    speaker1_splitted = speaker1.split()
    if not len(speaker1_splitted[0]) == 2 or not speaker1_splitted[0][1] == "'" or not len(speaker1_splitted) > 1:
        return speaker1
    for speaker2 in speakers_dict[speaker1_splitted[-1]]:
        speaker2_splitted = speaker2.split()
        if len(speaker2_splitted[0]) >= 2 and speaker2_splitted[0][1] != "'" and len(speaker2_splitted) > 1 and speaker1_splitted[0][0] == speaker2_splitted[0][0] and speaker1_splitted[-1] == speaker2_splitted[-1]:
            return speaker2
    return speaker1

#Input: path to a folder
#Output: file names, paths, and contents for all .docx files in the folder
def read_files(folder):
    file_names = [file for file in os.listdir(folder) if file[-5:] == '.docx']
    file_paths = {filename: os.path.join(folder, filename) for filename in file_names}
    file_contents = {file_name: Document(file_path) for file_name, file_path in file_paths.items()}
    return file_names, file_paths, file_contents

def jsonl_make(protocols, file):
    with open(file, 'w', encoding = 'utf-8') as jsonl_file:
        for protocol in protocols:
            for speaker, sentences in protocol.sentences.items():
                speaker_f = speaker_full_name(speaker)
                for sentence in sentences:
                    sentence_data = {
                        "protocol_name": sentence.protocol_name,
                        "knesset_number": sentence.keneset,
                        "protocol_type": sentence.protocol_type,
                        "protocol_number": sentence.protocol_no,
                        "speaker_name": speaker_f,
                        "sentence_text": sentence.text
                    }
                    jsonl_file.write(json.dumps(sentence_data, ensure_ascii=False) + '\n')

import time
def main():
    start_t = time.time()
    #if len(sys.argv) != 3:
    #    print("Error: Incorrect # of arguments.\n")
    #    sys.exit(1)
    #else:
    #    print("Creating the corpus ..\n")
    #folder_path = sys.argv[1] 
    folder_path = "protocol_for_hw1" 
    #file = sys.argv[2] 
    file = "corpus3.jsonl"
    file_names, file_paths, file_contents = read_files(folder_path)
    protocols = []
    for file_name in sorted(file_names): 
        keneset_no, protocol_type = extract_metada_from_name(file_name)
        protocol_no = extract_metada_from_content(file_contents[file_name])
        protocol = Protocol(file_name, keneset_no, protocol_type, protocol_no)
        protocols.append(protocol)
        extract_relevant_text(file_contents[file_name], protocol)
        protocol.check()
    #jsonl_make(protocols, file)
    end_t = time.time()
    print(end_t - start_t)

if __name__ == "__main__":
    main()
