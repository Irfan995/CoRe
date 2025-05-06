BERT_MODEL_NAMES = {
    'base-uncased': 'bert-base-uncased',
    'large-uncased': 'bert-large-uncased',
    'large-uncased-whole-word-masking': 'bert-large-uncased-whole-word-masking',
    'large-cased-whole-word-masking': 'bert-large-cased-whole-word-masking',
    'base-cased': 'bert-base-cased',
    'large-cased': 'bert-large-cased'
}

SBERT_MODEL_NAMES = {
    'all-MiniLM-L6-v2': {'name': 'all-MiniLM-L6-v2', 'dimension': 384},
    'all-MiniLM-L12-v2': {'name': 'all-MiniLM-L12-v2', 'dimension': 384},
    'all-mpnet-base-v2': {'name': 'all-mpnet-base-v2', 'dimension': 768},
    'multi-qa-mpnet-base-dot-v1': {'name': 'multi-qa-mpnet-base-dot-v1', 'dimension': 768},
    'all-distilroberta-v1': {'name': 'all-distilroberta-v1', 'dimension': 768},
    'multi-qa-distilbert-cos-v1': {'name': 'multi-qa-distilbert-cos-v1', 'dimension': 768},
    'multi-qa-MiniLM-L6-cos-v1': {'name': 'multi-qa-MiniLM-L6-cos-v1', 'dimension': 384},
    'paraphrase-MiniLM-L3-v2': {'name': 'paraphrase-MiniLM-L3-v2', 'dimension': 384},
    'paraphrase-albert-small-v2': {'name': 'paraphrase-albert-small-v2', 'dimension': 768}
}