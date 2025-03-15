import streamlit as st
import pymongo
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spellchecker import SpellChecker
from bson import ObjectId  # <-- Import ObjectId from bson

# MongoDB connection
client = pymongo.MongoClient()
db = client['CPP_Biology']
faculty_collection = db['FacultyInfo']
inverted_index_collection = db['InvertedIndex']
embeddings_collection = db['Embeddings']

# Load SpaCy model for lemmatization
nlp = spacy.load("en_core_web_lg")

# Initialize spell checker
spell = SpellChecker()

# Load TF-IDF vectorizer
TFIDF_PKL_FILE = "tfidf_vectorizer.pkl"
def load_vectorizer():
    try:
        with open(TFIDF_PKL_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Vectorizer file not found. Please generate the index first.")
        return None
vectorizer = load_vectorizer()

# Function to lemmatize query
def lemmatize_query(query):
    doc = nlp(query)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

# Function to check spelling
def check_spelling(query):
    words = query.split()
    misspelled = spell.unknown(words)
    return {word: spell.correction(word) for word in misspelled}

# Search function using TF-IDF
def searchWithTFIDF(query_sentence):
    query_vector, query_terms = process_query(query_sentence)
    candidate_docs = collect_candidate_documents(query_terms)
    return compute_similarity_scores(query_vector, candidate_docs)

# Process query
def process_query(query_sentence):
    query_vector = vectorizer.transform([query_sentence]).toarray()
    query_terms = vectorizer.inverse_transform(query_vector)[0]
    return query_vector, query_terms

# Collect candidate documents
def collect_candidate_documents(query_terms):
    candidate_docs = {}
    for term in query_terms:
        term_entry = inverted_index_collection.find_one({"term": term})
        if term_entry:
            for doc in term_entry['documents']:
                doc_id = doc['document_id']
                candidate_docs[doc_id] = candidate_docs.get(doc_id, 0) + doc['tfidf_score']
    return candidate_docs

# Compute similarity scores
def compute_similarity_scores(query_vector, candidate_docs):
    results = []
    for doc_id, score in candidate_docs.items():
        embedding_entry = embeddings_collection.find_one({"document_id": doc_id})
        if embedding_entry:
            embedding = np.array(embedding_entry["tfidf"])
            similarity = cosine_similarity(query_vector, embedding.reshape(1, -1))[0][0]
            results.append(fetch_document_details(doc_id, similarity))
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Fetch faculty details
def fetch_document_details(doc_id, similarity):
    faculty_details = faculty_collection.find_one({"_id": ObjectId(doc_id)})  # Corrected line
    return {
        "name": faculty_details.get("faculty_name", "N/A"),
        "similarity": round(similarity, 2),
        "url": faculty_details.get("profile_url", "N/A"),
        "summary": faculty_details.get("summary", "N/A")
    }

# Streamlit UI
st.title("🔍 Biology Faculty Search Engine")
query = st.text_input("Enter your search query:")

if query:
    # Spell check suggestions
    corrections = check_spelling(query)
    suggested_query = query

    if corrections:
        for word, suggestion in corrections.items():
            if suggestion:  # Ensure suggestion is not None
                suggested_query = suggested_query.replace(word, suggestion)

        # Only show the suggestion if it's different from the original query
        if suggested_query != query:
            st.warning(f"Did you mean: **{suggested_query}**?")
            if st.button("Use suggested query"):
                query = suggested_query

    
    # Process query and search
    lemmatized_query = lemmatize_query(query)
    results = searchWithTFIDF(lemmatized_query)
    
    if not results:
        st.error("No relevant results found for your query.")
    else:
        st.subheader("Search Results")
        for result in results:
            st.write(f"**{result['name']}**")
            st.write(f"Similarity Score: {result['similarity']}")
            st.write(f"[Profile Link]({result['url']})")
            st.write(f"Summary: {result['summary']}")
            st.markdown("---")
