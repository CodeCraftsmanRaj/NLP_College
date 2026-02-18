import numpy as np
from collections import defaultdict, Counter
from hmmlearn import hmm

corpus = [
    [("the", "DET"), ("park", "NOUN"), ("is", "VERB"), ("beautiful", "ADJ")],
    [("they", "PRON"), ("park", "VERB"), ("cars", "NOUN")],
    [("a", "DET"), ("park", "NOUN"), ("opens", "VERB")],
    [("people", "NOUN"), ("park", "VERB"), ("outside", "ADV")]
]

transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()
start_counts = Counter()

for sentence in corpus:
    first_tag = sentence[0][1]
    start_counts[first_tag] += 1
    for i, (word, tag) in enumerate(sentence):
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1
        if i > 0:
            prev_tag = sentence[i-1][1]
            transition_counts[prev_tag][tag] += 1

tags = sorted(tag_counts.keys())
vocab = sorted({word for sentence in corpus for word, _ in sentence})

tag_index = {tag: i for i, tag in enumerate(tags)}
word_index = {word: i for i, word in enumerate(vocab)}

N = len(tags)
V = len(vocab)

start_prob = np.zeros(N)
transition_matrix = np.zeros((N, N))
emission_matrix = np.zeros((N, V))

for tag in tags:
    start_prob[tag_index[tag]] = start_counts[tag] / sum(start_counts.values())

for prev_tag in tags:
    total = sum(transition_counts[prev_tag].values())
    if total == 0:
        transition_matrix[tag_index[prev_tag]] = np.ones(N)
    else:
        for tag in tags:
            transition_matrix[tag_index[prev_tag]][tag_index[tag]] = transition_counts[prev_tag][tag]

transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

for tag in tags:
    for word in vocab:
        emission_matrix[tag_index[tag]][word_index[word]] = emission_counts[tag][word]

emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

def manual_probability(sentence, tag_sequence):
    prob = start_prob[tag_index[tag_sequence[0]]] * emission_matrix[tag_index[tag_sequence[0]]][word_index[sentence[0]]]
    for i in range(1, len(sentence)):
        prob *= transition_matrix[tag_index[tag_sequence[i-1]]][tag_index[tag_sequence[i]]]
        prob *= emission_matrix[tag_index[tag_sequence[i]]][word_index[sentence[i]]]
    return prob

test_sentence = ["they", "park", "cars"]

path1 = ["PRON", "VERB", "NOUN"]
path2 = ["PRON", "NOUN", "NOUN"]

manual_prob1 = manual_probability(test_sentence, path1)
manual_prob2 = manual_probability(test_sentence, path2)
manual_final = max(manual_prob1, manual_prob2)

model = hmm.CategoricalHMM(n_components=N, init_params="")
model.startprob_ = start_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

obs = np.array([[word_index[w]] for w in test_sentence])
log_prob = model.score(obs)
library_prob = np.exp(log_prob)

print("Manual Path 1 Probability:", manual_prob1)
print("Manual Path 2 Probability:", manual_prob2)
print("Manual Final Probability:", manual_final)
print("Library Probability:", library_prob)
