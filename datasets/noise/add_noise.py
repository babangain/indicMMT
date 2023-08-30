import random
from tqdm import tqdm
random.seed(35)

colors = ['orange', 'green', 'red', 'white', 'black', 'pink', 'blue', 'purple', 'tan', 'grey', 'gray', 'yellow', 'gold', 'golden', 'dark', 'brown', 'silver']
articles= ["a", "an", "the"]
vowels = ["a","e","i","o","u"]
input_file_prefix="../dataset/"
for file in ['train/en-hi/train.en', 'valid/valid.en','test/test.en','challenge/challenge.en']:
    f = open(input_file_prefix+ file)
    g = open(file,'w')
    data = f.readlines()
    for sentence in tqdm(data):
        sentence = sentence.strip().split()
        output = sentence
        idx_to_remove = []
        for idx, i in enumerate(sentence):
            removed = False
            changed = False
            if i.lower() in articles:
                if random.random() > 0.8:
                    idx_to_remove.append(idx)
                    removed = True

            if not removed and (i.isalpha()):
                chars = list(i)
                new_char = chars[0]
                for x in range(1, len(chars)):
                    if chars[x].lower() in vowels: #drop vowels
                        if random.random() > 0.9:
                            changed = True
                            continue
                    if chars[x] == chars[x-1]: # drop two consecutive chars
                        if random.random() > 0.8:
                            changed = True
                            continue
                    new_char = new_char + chars[x]
                output[idx] = new_char

        for y in sorted(idx_to_remove, reverse=True):
            del sentence[y]

        g.write(" ".join(sentence))
        g.write("\n")
