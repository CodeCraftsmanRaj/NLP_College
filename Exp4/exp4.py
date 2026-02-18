import numpy as np
import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
from itertools import product
from hmmlearn import hmm

nltk.download('brown')
nltk.download('universal_tagset')

# Print full matrices without truncation
np.set_printoptions(threshold=np.inf, linewidth=200)

# Take ONLY ONE sentence from Brown corpus
# tagged_sentence = brown.tagged_sents(tagset='universal')[100]

tagged_sentence = [('Daniel', 'NOUN'), ('personally', 'ADV'), ('likes', 'VERB'), ('to', 'PRT'), ('read', 'VERB')]

print("\nSentence Used For Computation:\n")
print([word.lower() for word, tag in tagged_sentence])
print("\nTagged Version:\n")
print(tagged_sentence)

transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()
start_counts = Counter()

if len(tagged_sentence) != 0:

    first_tag = tagged_sentence[0][1]
    start_counts[first_tag] += 1

    for i, (word, tag) in enumerate(tagged_sentence):
        word = word.lower()
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

        if i > 0:
            prev_tag = tagged_sentence[i - 1][1]
            transition_counts[prev_tag][tag] += 1

tags = sorted(tag_counts.keys())
vocab = sorted({word.lower() for word, _ in tagged_sentence})

tag_index = {tag: i for i, tag in enumerate(tags)}
word_index = {word: i for i, word in enumerate(vocab)}

N = len(tags)
V = len(vocab)

start_prob = np.zeros(N)
transition_matrix = np.zeros((N, N))
emission_matrix = np.zeros((N, V))

for tag in tags:
    start_prob[tag_index[tag]] = start_counts[tag]

start_prob = start_prob / start_prob.sum()

for prev_tag in tags:
    for tag in tags:
        transition_matrix[tag_index[prev_tag]][tag_index[tag]] = transition_counts[prev_tag][tag]

if transition_matrix.sum(axis=1).any():
    transition_matrix = np.divide(
        transition_matrix,
        transition_matrix.sum(axis=1, keepdims=True),
        where=transition_matrix.sum(axis=1, keepdims=True) != 0
    )

# Emission matrix
for tag in tags:
    for word in vocab:
        emission_matrix[tag_index[tag]][word_index[word]] = emission_counts[tag][word]

if emission_matrix.sum(axis=1).any():
    emission_matrix = np.divide(
        emission_matrix,
        emission_matrix.sum(axis=1, keepdims=True),
        where=emission_matrix.sum(axis=1, keepdims=True) != 0
    )

print("\nFull Transition Matrix:\n")
print(transition_matrix)

print("\nFull Emission Matrix:\n")
print(emission_matrix)

# Sentence words only
sentence = [word.lower() for word, tag in tagged_sentence]

def get_possible_tags(word):
    return [tag for tag in tags if emission_counts[tag][word] > 0]

def manual_probability(sentence, tag_sequence):
    prob = start_prob[tag_index[tag_sequence[0]]] * \
           emission_matrix[tag_index[tag_sequence[0]]][word_index[sentence[0]]]

    for i in range(1, len(sentence)):
        prob *= transition_matrix[tag_index[tag_sequence[i - 1]]][tag_index[tag_sequence[i]]]
        prob *= emission_matrix[tag_index[tag_sequence[i]]][word_index[sentence[i]]]

    return prob

possible_tags_per_word = [get_possible_tags(word) for word in sentence]
all_paths = list(product(*possible_tags_per_word))

best_path = None
best_prob = 0

for path in all_paths:
    prob = manual_probability(sentence, path)
    if prob > best_prob:
        best_prob = prob
        best_path = path

model = hmm.CategoricalHMM(n_components=N, init_params="")
model.startprob_ = start_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

obs = np.array([[word_index[w]] for w in sentence])

log_prob_score = model.score(obs)
library_prob = np.exp(log_prob_score)

log_prob_decode, state_sequence = model.decode(obs)
library_path = [tags[state] for state in state_sequence]

print("\nManual Best Path:", best_path)
print("Manual Probability:", best_prob)

print("\nLibrary Best Path:", library_path)
print("Library Probability:", library_prob)
