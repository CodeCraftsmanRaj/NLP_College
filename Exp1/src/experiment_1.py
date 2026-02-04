import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import os

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

def run_aim_1():
    print("\n" + "="*40)
    print(" AIM 1: NLTK Basic Analysis & FreqDist")
    print("="*40)

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

    tokens = word_tokenize(text_content)
    print(f"\nTotal Tokens before removal: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")

    print("\n" + "-"*30)
    print(" STOP WORD REMOVAL (MANUAL)")
    print("-" * 30)
    
    user_input = input("Enter stop words to remove (separated by space, e.g., 'the is and of'): ")
    
    stop_words_list = [w.strip().lower() for w in user_input.split()]
    
    print(f"Stop words to remove: {stop_words_list}")

    filtered_tokens = []
    
    for token in tokens:
        if token.lower() not in stop_words_list:
            filtered_tokens.append(token)

    print(f"\nTotal Tokens after removal: {len(filtered_tokens)}")
    print(f"First 10 filtered tokens: {filtered_tokens[:10]}")

    fdist = FreqDist(filtered_tokens)
    print("\nTop 5 Most Common Words (After Cleanup):")
    print(fdist.most_common(5))

    print("\nDisplaying Frequency Plot... (Close the plot window to continue)")
    plt.figure(figsize=(10, 7))  
    plt.title("Word Frequency Distribution (Stop Words Removed)")
    fdist.plot(20, cumulative=False)
    plt.show() 

def run_aim_2():
    print("\n" + "="*40)
    print(" AIM 2: Morphological Analysis (Add-Delete Table)")
    print("="*40)

    user_root = input("Enter Source/Root Word (e.g., teach): ").strip()
    user_final = input("Enter Final Form Word (e.g., teaches): ").strip()
    
    results = []

    if user_root and user_final:
        delete_rule = "-"
        add_rule = "-"
        
        common_len = 0
        min_len = min(len(user_root), len(user_final))
        
        for i in range(min_len):
            if user_root[i] == user_final[i]:
                common_len += 1
            else:
                break
        
        del_str = user_root[common_len:]
        add_str = user_final[common_len:]

        if del_str:
            delete_rule = del_str
        if add_str:
            add_rule = add_str

        results.append({
            "Source (Root)": user_root,
            "Final Form": user_final,
            "Delete": delete_rule,
            "Add": add_rule,
            "Number": "User-Input",
            "Gender": "-",
            "Case": "-"
        })

    df = pd.DataFrame(results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\nFinal Add-Delete Table:")
    print(df)

if __name__ == "__main__":
    run_aim_1()
    run_aim_2()