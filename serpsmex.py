import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from urllib.parse import urlencode

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to get top 10 Google search results using CSE API
def get_top_search_results(api_key, cse_id, keyword, country, num_results=10):
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': keyword,
        'gl': country,
        'num': num_results
    }
    search_url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
    response = requests.get(search_url)
    results = response.json()
    return results.get('items', [])

# Function to scrape and clean webpage content
def scrape_webpage_content(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ')
            text = re.sub(r'[^a-zA-Z\sñáéíóúüÁÉÍÓÚÜÑ]', '', text)
            return text.lower()
        else:
            st.write(f"Error al extraer {url}: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.write(f"Error al extraer {url}: {e}")
        return None

# Function to preprocess text using NLTK
def preprocess_text(text, stopwords):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords]
    return tokens

# Function to compute TF-IDF scores
def compute_tfidf(corpus, stopwords):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    tfidf_scores = X.toarray()
    terms = vectorizer.get_feature_names_out()
    return terms, tfidf_scores

# Convert TF-IDF score to recommended frequency based on the average word count
def convert_tfidf_to_frequency(tfidf_score, total_word_count, avg_word_count, scaling_factor=100):
    frequency = int(tfidf_score * avg_word_count / total_word_count * scaling_factor)
    return max(frequency, 1)  # Ensure at least one occurrence

# Streamlit app
def main():
    st.title("Analizador de TF-IDF de Google Search")
    
    # Input for Google CSE API Key and Search Engine ID
    api_key = st.text_input("Introduce tu clave de API de Google:")
    cse_id = st.text_input("Introduce tu ID de motor de búsqueda personalizado (CSE):")
    
    # Input for keyword
    keyword = st.text_input("Introduce una palabra clave:")
    
    # Dropdown for selecting country
    country = st.selectbox("Selecciona un país", options=['us', 'es', 'fr', 'de', 'it','mex'])
    
    if st.button("Analizar"):
        # Validation
        if not api_key or not cse_id:
            st.error("Por favor, introduce tanto la clave de API de Google como el ID de motor de búsqueda personalizado (CSE).")
            return
        
        if not keyword:
            st.error("Por favor, introduce una palabra clave.")
            return
        
        # Get the top 10 search results
        results = get_top_search_results(api_key, cse_id, keyword, country)
        
        if not results:
            st.write("No se encontraron resultados.")
            return
        
        st.write("Los 10 mejores resultados:")
        for result in results:
            st.write(f"**Título**: {result['title']}")
            st.write(f"**Descripción**: {result['snippet']}")
            st.write(f"**URL**: {result['link']}")
            st.write("---")
        
        # Scrape content from each result and preprocess it
        corpus = []
        word_counts = []
        valid_urls = []
        for result in results:
            url = result['link']
            text = scrape_webpage_content(url)
            if text:
                corpus.append(text)
                word_counts.append(len(text.split()))  # Store word count for each page
                valid_urls.append(url)
        
        if not corpus:
            st.write("No hay contenido válido para procesar.")
            return
        
        # Calculate the average word count of the top 3 results
        if len(word_counts) >= 3:
            avg_word_count = sum(word_counts[:3]) // 3
        else:
            avg_word_count = sum(word_counts) // len(word_counts)  # Fallback if fewer than 3 results
        
        # Perform TF-IDF analysis
        terms, tfidf_scores = compute_tfidf(corpus, stopwords.words('spanish'))
        total_word_count = sum(len(text.split()) for text in corpus)
        
        # Prepare recommendations for download
        recommendations = []
        for i, url in enumerate(valid_urls):
            num_terms = len(tfidf_scores[i])
            sorted_indices = tfidf_scores[i].argsort()[::-1][:num_terms]
            
            for index in sorted_indices:
                term = terms[index]
                score = tfidf_scores[i][index]
                frequency = convert_tfidf_to_frequency(score, total_word_count, avg_word_count)
                recommendations.append({
                    'Palabra clave': term,
                    'Puntuación TF-IDF': score,
                    'Frecuencia recomendada': frequency,
                    'URL de origen': url
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(recommendations)
        
        # Display recommendations
        st.write("### Recomendaciones de TF-IDF")
        st.dataframe(df)
        
        # Provide download option
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="Descargar recomendaciones de TF-IDF en CSV",
            data=csv,
            file_name="recomendaciones_tfidf.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
