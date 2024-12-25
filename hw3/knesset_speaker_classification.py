import json
import random 
import numpy as np
random.seed(42)
np.random.seed(42)

#A method that extracts the json lines from a JSONL file (section 1)
def json_lines_extract(file):
    with open(file, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# A method used for extracting the top two speakers (section 1)
# def top_two_speakers(json_lines):
#     speakers = {}
#     for line in json_lines:
#         speaker = line['speaker_name']
#         if speaker in speakers:
#             speakers[speaker] += 1
#         else:
#             speakers[speaker] = 1
#     sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
#     return sorted_speakers[:2]

# A method that splits the sentences according to the speaker (section 1.2)
def split_data_by_speaker(json_lines, speaker1, speaker2):
    speaker1_data = []
    speaker2_data = []
    others_data = []
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
                others_data.append(line)
    return speaker1_data, speaker2_data, others_data

def main():
    file = 'knesset_corpus.jsonl'
    json_lines = json_lines_extract(file)

    #lines used for extracting the top two speakers in section 1
    #top_speakers = top_two_speakers(json_lines)
    #print(top_speakers)

    #split the data according to the speaker (section 1.2)
    rivlin_full_data, burg_full_data, others_full_data = split_data_by_speaker(json_lines, "ראובן ריבלין", "א' בורג")
    
    #classes balancing (section 2)
    class_count = min(len(rivlin_full_data), len(burg_full_data), len(others_full_data))
    rivlin_data = random.sample(rivlin_full_data, class_count)
    burg_data = random.sample(burg_full_data, class_count)
    others_data = random.sample(others_full_data, class_count)

    # #printing the count of sentences of each class before and after the down sampling (section 2)
    # print('Sentences count of each class before the down sampling:')
    # print('rivlin_sentences:', len(rivlin_full_data))
    # print('burg_sentences:', len(burg_full_data))
    # print('others_sentences:', len(others_full_data))
    # print('Sentences count of each class after the down sampling:')
    # print('rivlin_sentences:', len(rivlin_data))
    # print('burg_sentences:', len(burg_data))
    # print('others_sentences:', len(others_data))

if __name__ == '__main__':
    main()