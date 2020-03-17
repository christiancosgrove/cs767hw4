import pickle
import os
from parlai.core.dict import DictionaryAgent

path = 'dat/MovieTriples_Dataset.tar'

with open(os.path.join(path, 'Training.dict.pkl'), 'rb') as data_file:
    dictionary = pickle.load(data_file)


parlai_dict = DictionaryAgent({'vocab_size' : 10004})

dictionary = sorted(dictionary, key=lambda x: x[1])
print(dictionary[:10])

for word in dictionary:
    # print(word[0])
    parlai_dict.add_to_dict([word[0]])
    parlai_dict.freq[word[0]] = word[2]
    # print(word)

# print(parlai_dict)
# parlai_dict.add_to_dict(['hello'])

parlai_dict.save('test_hred.dict', sort=True)
