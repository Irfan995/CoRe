import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_excel(r'datasets\classroom_short_commands.xlsx')

unique_label = data['Context'].unique()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase and remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text.lower())
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def print_topics(model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def calculate_coherence(lda_model, tokenized_docs, dictionary):
    # Compute Coherence Score
    coherence_model = CoherenceModel(
        model=lda_model,  # Gensim-compatible LDA model
        texts=tokenized_docs,  # Preprocessed tokenized documents
        dictionary=dictionary,  # Gensim dictionary
        coherence='c_v'  # Coherence metric (default: 'c_v')
    )
    return coherence_model.get_coherence()

def main():
    # Main loop with coherence calculation
    for label in unique_label:
        print(f"------------------- {label} -------------------")
        documents = data[data['Context'] == label]['Request']

        # Preprocess the documents
        tokenized_docs = [preprocess(doc) for doc in documents]

        # Create bigram and trigram models
        bigram = Phrases(tokenized_docs, min_count=2, threshold=10)
        trigram = Phrases(bigram[tokenized_docs], threshold=10)

        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        trigram_docs = [trigram_mod[bigram_mod[doc]] for doc in tokenized_docs]

        # Create Gensim dictionary and corpus
        dictionary = Dictionary(trigram_docs)
        corpus = [dictionary.doc2bow(doc) for doc in trigram_docs]

        # Join tokens into sentences
        trigram_sentences = [' '.join(doc) for doc in trigram_docs]

        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(trigram_sentences)

        # cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # print(cosine_sim_matrix)

        # Initialize LDA model
        n_topics = 2  # Adjust number of topics as needed
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=10)

        # Calculate coherence score
        coherence_score = calculate_coherence(lda_model, trigram_docs, dictionary)

        print(f"TF-IDF Matrix Shape: {len(corpus)} documents, {len(dictionary)} unique tokens")
        print(f"Coherence Score: {coherence_score}")

        # Print topics
        for idx, topic in lda_model.show_topics(formatted=False, num_words=10):
            print(f"Topic #{idx + 1}: " + " ".join([word for word, _ in topic]))

        # Fit LDA model
        # lda_model.fit(tfidf_matrix)

        # Extract topics for coherence calculation
        # feature_names = tfidf_vectorizer.get_feature_names_out()
        # topics = [
        #     [feature_names[i] for i in topic.argsort()[:-11:-1]]  # Top 10 terms per topic
        #     for topic in lda_model.components_
        # ]

        # # Convert topics to Gensim-compatible format
        # gensim_lda_topics = [[dictionary.token2id[word] for word in topic if word in dictionary.token2id] for topic in topics]

        # Calculate coherence score
        # coherence_score = calculate_coherence(gensim_lda_topics, trigram_docs, dictionary)

        # print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
        # print_topics(lda_model, tfidf_vectorizer)
        # print(f"Coherence Score: {coherence_score}")
        
if __name__ == "__main__":
    main()