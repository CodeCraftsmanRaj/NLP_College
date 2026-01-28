import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import os

# --- SETUP ---
# Updated: newer NLTK versions require 'punkt_tab' specifically
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True) # Keeping legacy just in case

def run_aim_1():
    print("\n" + "="*40)
    print(" AIM 1: NLTK Basic Analysis & FreqDist")
    print("="*40)

    # 1. Load Data
    file_path = os.path.join("data", "sample.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        print(f"Loaded text from {file_path}")
    else:
        print(f"File {file_path} not found. Using default text.")
        text_content = """
        Natural language processing (NLP) refers to the branch of computer science concerned with 
        giving computers the ability to understand text and spoken words in much the same way human 
        beings can. NLP drives computer programs that translate text from one language to another, 
        respond to spoken commands, and summarize large volumes of text rapidly.
        """

    # 2. Tokenization
    # word_tokenize relies on sentence tokenization internally, which needs punkt_tab
    tokens = word_tokenize(text_content)
    print(f"\nTotal Tokens: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")

    # 3. Frequency Distribution
    fdist = FreqDist(tokens)
    print("\nTop 5 Most Common Words:")
    print(fdist.most_common(5))

    # 4. Plotting
    print("\nDisplaying Frequency Plot... (Close the plot window to continue)")
    plt.figure(figsize=(10, 5))
    plt.title("Word Frequency Distribution")
    fdist.plot(20, cumulative=False)
    plt.show() # This will open a popup window

def run_aim_2():
    print("\n" + "="*40)
    print(" AIM 2: Morphological Analysis (Add-Delete Table)")
    print("="*40)

    # Data defined manually based on the experiment theory
    data = [
        # Root, Final, Number, Gender, Case
        ("Teach", "Teaches", "Singular", "-", "Present"),
        ("Teach", "Taught", "Singular", "-", "Past"),
        ("Play", "Played", "-", "-", "Past"),
        ("Play", "Playing", "-", "-", "Continuous"),
        ("Push", "Pushes", "Singular", "-", "Present")
    ]

    results = []

    for root, final, num, gen, case in data:
        delete_rule = "-"
        add_rule = "-"

        # Logic for Teach -> Teaches / Push -> Pushes
        if final.startswith(root):
            suffix = final[len(root):]
            if suffix:
                add_rule = suffix
        
        # Logic for Teach -> Taught (Irregular)
        elif root == "Teach" and final == "Taught":
            delete_rule = "Ch"  # Deleting 'ch' from Teach leaves 'Tea'
            add_rule = "ught"   # Adding 'ught' makes 'Taught'
            # We strictly follow the theory manual mapping here:
            delete_rule = "Ch"
            add_rule = "aught"

        results.append({
            "Source (Root)": root,
            "Final Form": final,
            "Delete": delete_rule,
            "Add": add_rule,
            "Number": num,
            "Gender": gen,
            "Case": case
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Adjust formatting for terminal output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df)

if __name__ == "__main__":
    run_aim_1()
    run_aim_2()