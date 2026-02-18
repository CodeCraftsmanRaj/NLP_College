from collections import Counter, defaultdict

corpus = [
    "I drink coffee",
    "I drink tea",
    "You drink coffee"
]

sentences = []
for s in corpus:
    sentences.append(["<s>"] + s.lower().split() + ["</s>"])

unigram_counts = Counter()
bigram_counts = defaultdict(Counter)

for sent in sentences:
    for i in range(len(sent)):
        unigram_counts[sent[i]] += 1
        if i > 0:
            bigram_counts[sent[i-1]][sent[i]] += 1

vocab = list(unigram_counts.keys())
V = len(vocab)

print("\nUNIGRAM COUNTS")
for w in unigram_counts:
    print(w, ":", unigram_counts[w])

print("\nBIGRAM COUNTS")
for prev in bigram_counts:
    for curr in bigram_counts[prev]:
        print(f"({prev}, {curr}) :", bigram_counts[prev][curr])

print("\nBIGRAM PROBABILITIES (NO SMOOTHING)")
for prev in bigram_counts:
    for curr in bigram_counts[prev]:
        print(f"P({curr}|{prev}) =", bigram_counts[prev][curr] / unigram_counts[prev])

print("\nBIGRAM PROBABILITIES (ADD-ONE SMOOTHING)")
for prev in unigram_counts:
    for curr in vocab:
        print(
            f"P*({curr}|{prev}) =",
            (bigram_counts[prev][curr] + 1) / (unigram_counts[prev] + V)
        )

def sentence_probability(sentence, smoothing=False):
    tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
    prob = 1
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        curr = tokens[i]
        if smoothing:
            p = (bigram_counts[prev][curr] + 1) / (unigram_counts[prev] + V)
            print(f"P*({curr}|{prev}) =", p)
        else:
            if bigram_counts[prev][curr] == 0:
                print(f"P({curr}|{prev}) = 0")
                return 0
            p = bigram_counts[prev][curr] / unigram_counts[prev]
            print(f"P({curr}|{prev}) =", p)
        prob *= p
    return prob

sentence = input("\nEnter a sentence: ")
threshold = float(input("Enter acceptance threshold: "))

print("\nWITHOUT SMOOTHING")
p1 = sentence_probability(sentence)
print("Final Probability =", p1)
if p1 >= threshold:
    print("Conclusion: ACCEPTED")
else:
    print("Conclusion: NOT ACCEPTED")

print("\nWITH ADD-ONE SMOOTHING")
p2 = sentence_probability(sentence, smoothing=True)
print("Final Probability =", p2)
if p2 >= threshold:
    print("Conclusion: ACCEPTED")
else:
    print("Conclusion: NOT ACCEPTED")
