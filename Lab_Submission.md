Aim:
Perform basic corpus analysis using NLTK, remove stop words, visualize word frequency, compare NLTK vs spaCy, and analyze morphological changes using an Add-Delete table.

Problem statement:
Given a text paragraph, tokenize it, remove stop words using both library and manual lists, show stop words found, compute frequency distribution with visualization, and document a comparative study of NLTK and spaCy. Also, build an Add-Delete table for morphological analysis using user-supplied source and final forms.

Theory:
Corpus is a structured collection of text used for analysis. Stop words are high-frequency function words (e.g., “the”, “is”) that often carry little semantic weight in many NLP tasks. NLTK provides tokenization, stop word lists, and frequency analysis tools. spaCy provides fast, production-ready NLP pipelines (tokenization, POS, NER, parsing). Morphological analysis studies word formation; the Add-Delete table records the suffix/prefix changes between a root and its final form.

Algorithm:
1. Read a paragraph from user input; if empty, load data/sample.txt.
2. Tokenize the text using NLTK.
3. Load NLTK stop words and a manual stop word list.
4. Identify stop words present in the paragraph.
5. Display the paragraph after stop word removal using (a) NLTK list and (b) manual list.
6. Compute frequency distribution on cleaned tokens and visualize it.
7. Build a comparison table between NLTK and spaCy.
8. Accept source and final forms from the user.
9. Compute Add/Delete strings via longest common prefix.
10. Collect Number, Gender, Case, and Tense inputs and display the Add-Delete table.

Program:
The implementation is in src/experiment_1.py and invoked by main.py. It performs tokenization, stop word analysis, frequency visualization (saved to output/freq_dist.png), prints the NLTK vs spaCy comparison table, and generates the Add-Delete table from user input.

Output:
- Tokens before/after stop word removal
- Stop words present in the paragraph
- Paragraph without stop words (NLTK and manual)
- Top word frequencies and visualization
- Comparative study table (also saved as output/nltk_vs_spacy.csv)
- Add-Delete table with Number, Gender, Case, and Tense

Questions to be answered:
1) What is corpus, stop words?
A corpus is a large, structured collection of texts used to train or evaluate NLP systems. Stop words are very common function words (e.g., “the”, “is”, “and”) that are often removed to reduce noise in tasks like search or topic modeling.

2) What is normalization in NLP? How does it work? Why is it important?
Normalization transforms text into a consistent form (e.g., lowercasing, removing punctuation, expanding contractions, stemming/lemmatization). It reduces variation so that semantically similar tokens are treated uniformly, improving matching, indexing, and model performance.

3) Describe different ambiguities in NLP with example.
- Lexical ambiguity: A word has multiple meanings (e.g., “bank” = river bank or financial bank).
- Syntactic ambiguity: Multiple parse structures (e.g., “I saw the man with a telescope.”).
- Semantic ambiguity: Multiple interpretations after parsing (e.g., “Visiting relatives can be boring.”).
- Pragmatic ambiguity: Meaning depends on context or speaker intent (e.g., “Can you open the window?” as a request).

4) What is WordNet and its relevance?
WordNet is a lexical database grouping words into synonym sets (synsets) with semantic relations (hypernyms, hyponyms, etc.). It is useful for semantic similarity, word sense disambiguation, and enriching NLP features.

Conclusion:
The experiment demonstrates basic corpus analysis with NLTK, stop word removal, word frequency visualization, and a comparative study of NLTK vs spaCy. Morphological analysis using the Add-Delete table captures word formation patterns. These steps provide foundational skills for NLP preprocessing and analysis.
