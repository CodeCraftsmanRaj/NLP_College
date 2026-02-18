import numpy as np
import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
from itertools import product
from hmmlearn import hmm

nltk.download('brown')
nltk.download('universal_tagset')

tagged_sentences = list(brown.tagged_sents(tagset='universal'))[:5000]

transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()
start_counts = Counter()

for sentence in tagged_sentences:
    if len(sentence) == 0:
        continue

    first_tag = sentence[0][1]
    start_counts[first_tag] += 1

    for i, (word, tag) in enumerate(sentence):
        word = word.lower()
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

        if i > 0:
            prev_tag = sentence[i - 1][1]
            transition_counts[prev_tag][tag] += 1

tags = sorted(tag_counts.keys())
vocab = sorted({word.lower() for sentence in tagged_sentences for word, _ in sentence})

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
    for tag in tags:
        transition_matrix[tag_index[prev_tag]][tag_index[tag]] = transition_counts[prev_tag][tag]

transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

for tag in tags:
    for word in vocab:
        emission_matrix[tag_index[tag]][word_index[word]] = emission_counts[tag][word]

emission_matrix = emission_matrix / emission_matrix.sum(axis=1, keepdims=True)

print("Transition Matrix:\n")
print(transition_matrix)

print("\nEmission Matrix:\n")
print(emission_matrix)

def get_possible_tags(word):
    word = word.lower()
    return [tag for tag in tags if emission_counts[tag][word] > 0]

def manual_probability(sentence, tag_sequence):
    prob = start_prob[tag_index[tag_sequence[0]]] * emission_matrix[tag_index[tag_sequence[0]]][word_index[sentence[0]]]

    for i in range(1, len(sentence)):
        prob *= transition_matrix[tag_index[tag_sequence[i - 1]]][tag_index[tag_sequence[i]]]
        prob *= emission_matrix[tag_index[tag_sequence[i]]][word_index[sentence[i]]]

    return prob

test_sentence = ["they", "park", "cars"]

possible_tags_per_word = [get_possible_tags(word) for word in test_sentence]
all_paths = list(product(*possible_tags_per_word))

best_path = None
best_prob = 0

for path in all_paths:
    prob = manual_probability(test_sentence, path)
    if prob > best_prob:
        best_prob = prob
        best_path = path

model = hmm.CategoricalHMM(n_components=N, init_params="")
model.startprob_ = start_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

obs = np.array([[word_index[w.lower()]] for w in test_sentence])

log_prob_score = model.score(obs)
library_prob = np.exp(log_prob_score)

log_prob_decode, state_sequence = model.decode(obs)
library_path = [tags[state] for state in state_sequence]

print("\nManual Best Path:", best_path)
print("Manual Probability:", best_prob)

print("\nLibrary Best Path:", library_path)
print("Library Probability:", library_prob)
