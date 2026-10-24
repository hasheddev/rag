import string

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
translation_table = str.maketrans("", "", string.punctuation)


def remove_punctuations(text: str):
    return text.translate(translation_table)


def tokenize_text(text: str, stopword_set: set[str]) -> list[str]:
    tokens = remove_punctuations(text).lower().split()
    if not tokens:
        return []
    return  [
        stemmer.stem(token) 
        for token in tokens 
        if token not in stopword_set
    ]


def tokenize_single_term(term: str, stopword_set: set[str]) -> str:
    tokens = tokenize_text(term, stopword_set)
    
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly 1 token for term '{term}', but got {len(tokens)}.")
    
    return tokens[0]