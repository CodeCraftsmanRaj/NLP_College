import nltk
import pandas as pd

from nltk.corpus import treebank
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer

# ---------------- DOWNLOAD REQUIRED DATA --------------------
nltk.download('treebank')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------- INPUT WORDS -------------------------------
words = [
    "studying", "universities", "fairly",
    "maximum", "provision", "company", "community"
]

# # ---------------- LOAD CORPUS -------------------------------
# sentences = treebank.sents()
# print("\n=========== ORIGINAL CORPUS (FIRST 2 SENTENCES) ===========\n")
# print(sentences[:2])

# ---------------- LIBRARY STEMMERS --------------------------
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

comparison = []

for word in words:
    comparison.append([
        word,
        porter.stem(word),
        snowball.stem(word),
        lancaster.stem(word),
        lemmatizer.lemmatize(word)
    ])

df = pd.DataFrame(
    comparison,
    columns=[
        "Original Word",
        "Porter Stemmer",
        "Snowball Stemmer",
        "Lancaster Stemmer",
        "Lemmatizer"
    ]
)

print("\n================ STEMMER vs LEMMATIZER =================\n")
print(df.to_string(index=False))


# ---------------- DOWNLOAD DATA ----------------
nltk.download('treebank')

# ---------------- CUSTOM PORTER STEMMER ----------------
class CustomPorterStemmer:
    def __init__(self):
        self.vowels = "aeiou"

    def is_consonant(self, word, i):
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            return i == 0 or not self.is_consonant(word, i - 1)
        return True

    def measure(self, word):
        m = 0
        i = 0
        length = len(word)

        while i < length and self.is_consonant(word, i):
            i += 1

        while i < length:
            while i < length and not self.is_consonant(word, i):
                i += 1
            if i < length:
                m += 1
            while i < length and self.is_consonant(word, i):
                i += 1

        return m

    def contains_vowel(self, word):
        return any(not self.is_consonant(word, i) for i in range(len(word)))

    def step1a(self, word):
        if word.endswith("sses"):
            return word[:-2]
        elif word.endswith("ies"):
            return word[:-2]
        elif word.endswith("ss"):
            return word
        elif word.endswith("s"):
            return word[:-1]
        return word

    def step1b(self, word):
        if word.endswith("eed"):
            if self.measure(word[:-3]) > 0:
                return word[:-1]
        elif word.endswith("ed"):
            stem = word[:-2]
            if self.contains_vowel(stem):
                return stem
        elif word.endswith("ing"):
            stem = word[:-3]
            if self.contains_vowel(stem):
                return stem
        return word

    def step1c(self, word):
        if word.endswith("y") and self.contains_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def stem(self, word):
        word = word.lower()
        word = self.step1a(word)
        word = self.step1b(word)
        word = self.step1c(word)
        return word


# ---------------- LOAD CORPUS ----------------
sentences = treebank.sents()

# Flatten words and keep only alphabetic tokens
original_words = [
    word.lower()
    for sentence in sentences
    for word in sentence
    if word.isalpha()
]

# Take first 30 words
original_words = original_words[:30]

# ---------------- APPLY CUSTOM STEMMER ----------------
custom_porter = CustomPorterStemmer()

stemmed_words = [custom_porter.stem(word) for word in original_words]

# ---------------- CREATE TABLE ----------------
df = pd.DataFrame({
    "Original Word": original_words,
    "Stemmed Word (Custom Porter)": stemmed_words
})

print("\n=========== CUSTOM PORTER STEMMER (NO LIBRARY) ===========\n")
print(df.to_string(index=False))