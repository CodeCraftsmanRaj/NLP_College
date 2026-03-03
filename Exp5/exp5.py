import numpy as np
import nltk
import time
import matplotlib.pyplot as plt
from nltk.corpus import brown
from collections import defaultdict, Counter
from itertools import product
from hmmlearn import hmm

# Download datasets
nltk.download('brown')
nltk.download('universal_tagset')

# Load corpus
tagged_sentences = brown.tagged_sents(tagset='universal')

transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()
start_counts = Counter()

# Build counts
for sent in tagged_sentences:
    if not sent:
        continue

    start_counts[sent[0][1]] += 1

    for i, (word, tag) in enumerate(sent):
        word = word.lower()
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

        if i > 0:
            prev_tag = sent[i - 1][1]
            transition_counts[prev_tag][tag] += 1

# Tags and vocabulary
tags = sorted(tag_counts.keys())
vocab = sorted({word for tag in emission_counts for word in emission_counts[tag]})

tag_index = {tag: i for i, tag in enumerate(tags)}
word_index = {word: i for i, word in enumerate(vocab)}

N = len(tags)
V = len(vocab)

start_prob = np.zeros(N)
transition_matrix = np.zeros((N, N))
emission_matrix = np.zeros((N, V))

# Start probabilities
for tag in tags:
    start_prob[tag_index[tag]] = start_counts[tag]

start_prob /= start_prob.sum()

# Transition matrix
for prev_tag in tags:
    for tag in tags:
        transition_matrix[tag_index[prev_tag]][tag_index[tag]] = transition_counts[prev_tag][tag]

transition_matrix = np.divide(
    transition_matrix,
    transition_matrix.sum(axis=1, keepdims=True),
    where=transition_matrix.sum(axis=1, keepdims=True) != 0
)

# Emission matrix
for tag in tags:
    for word in emission_counts[tag]:
        emission_matrix[tag_index[tag]][word_index[word]] = emission_counts[tag][word]

emission_matrix = np.divide(
    emission_matrix,
    emission_matrix.sum(axis=1, keepdims=True),
    where=emission_matrix.sum(axis=1, keepdims=True) != 0
)

# Test sentence
sentence = ["daniel", "likes", "reading"]

# Possible tags
def get_possible_tags(word):
    if word not in word_index:
        return []
    return [tag for tag in tags if emission_matrix[tag_index[tag]][word_index[word]] > 0]

possible_tags_per_word = [get_possible_tags(word) for word in sentence]

# Brute force probability
def manual_probability(sentence, tag_sequence):
    prob = start_prob[tag_index[tag_sequence[0]]] * \
           emission_matrix[tag_index[tag_sequence[0]]][word_index[sentence[0]]]

    for i in range(1, len(sentence)):
        prob *= transition_matrix[tag_index[tag_sequence[i - 1]]][tag_index[tag_sequence[i]]]
        prob *= emission_matrix[tag_index[tag_sequence[i]]][word_index[sentence[i]]]

    return prob

# ---------------- BRUTE FORCE ----------------
start_time = time.time()

all_paths = list(product(*possible_tags_per_word))
path_probs = []

for path in all_paths:
    prob = manual_probability(sentence, path)
    path_probs.append((path, prob))

path_probs.sort(key=lambda x: x[1], reverse=True)

brute_time = time.time() - start_time
brute_path = path_probs[0][0]
brute_prob = path_probs[0][1]

# ---------------- MANUAL VITERBI ----------------
def viterbi(sentence):
    T = len(sentence)
    viterbi_matrix = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    for s in range(N):
        viterbi_matrix[s, 0] = start_prob[s] * emission_matrix[s][word_index[sentence[0]]]

    for t in range(1, T):
        for s in range(N):
            probs = viterbi_matrix[:, t-1] * transition_matrix[:, s]
            best_prev = np.argmax(probs)

            viterbi_matrix[s, t] = probs[best_prev] * emission_matrix[s][word_index[sentence[t]]]
            backpointer[s, t] = best_prev

    best_last = np.argmax(viterbi_matrix[:, -1])
    best_path = [best_last]

    for t in range(T-1, 0, -1):
        best_last = backpointer[best_last, t]
        best_path.insert(0, best_last)

    best_tags = [tags[i] for i in best_path]
    best_prob = np.max(viterbi_matrix[:, -1])

    return best_tags, best_prob

start_time = time.time()

viterbi_path, viterbi_prob = viterbi(sentence)

viterbi_time = time.time() - start_time

# ---------------- HMMLEARN ----------------
model = hmm.CategoricalHMM(n_components=N, init_params="")
model.startprob_ = start_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

obs = np.array([[word_index[w]] for w in sentence])

start_time = time.time()

log_score = model.score(obs)
forward_prob = np.exp(log_score)

log_decode, states = model.decode(obs)
library_path = [tags[s] for s in states]

library_time = time.time() - start_time

# ---------------- RESULTS ----------------
print("Sentence:", sentence)

print("\nBrute Force Best Path:", brute_path)
print("Brute Force Probability:", brute_prob)
print("Brute Force Time:", brute_time)

print("\nViterbi Path:", viterbi_path)
print("Viterbi Probability:", viterbi_prob)
print("Viterbi Time:", viterbi_time)

print("\nHMMlearn Path:", library_path)
print("Forward Probability:", forward_prob)
print("Library Time:", library_time)

# ---------------- PLOTS ----------------

methods = ["Brute Force", "Manual Viterbi", "HMM Library"]
times = [brute_time, viterbi_time, library_time]
probs = [brute_prob, viterbi_prob, forward_prob]

# Time comparison
plt.figure()
plt.bar(methods, times)
plt.title("Execution Time Comparison")
plt.ylabel("Time (seconds)")
plt.xlabel("Method")
plt.show()

# Probability comparison
plt.figure()
plt.bar(methods, probs)
plt.title("Probability Comparison")
plt.ylabel("Probability")
plt.xlabel("Method")
plt.show()

# Tag comparison
print("\nTag Sequence Comparison")

print("Brute Force :", brute_path)
print("Viterbi     :", viterbi_path)
print("HMM Library :", library_path)