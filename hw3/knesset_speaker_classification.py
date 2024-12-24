import json

def json_lines_extract(file):
    with open(file, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# A method used for extracting the top two speakers in section 1 
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

def main():
    file = 'knesset_corpus.jsonl'
    json_lines = json_lines_extract(file)

    #lines used for extracting the top two speakers in section 1
    #top_speakers = top_two_speakers(json_lines)
    #print(top_speakers)

if __name__ == '__main__':
    main()