import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from collections import Counter

def load_text_files(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

def compute_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def compute_bow(texts):
    """Bow matrix create karo"""
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return bow_matrix, feature_names

def plot_word_cloud(word_freq, title, method='BoW', save_dir='figures'):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{method} Word Cloud: {title}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{title}_{method}_wordcloud.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_top_words(word_freq, title, top_n=20, method='BoW', save_dir='figures'):
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    color = 'lightcoral' if method == 'BoW' else 'skyblue'
    plt.barh(words, counts, color=color)
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} {method} Terms: {title}', fontsize=14)
    plt.xlabel('Frequency' if method == 'BoW' else 'TF-IDF Score')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{title}_{method}_topwords.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_word_frequency_distribution(word_freq, title, save_dir='figures'):
    freq_dist = list(word_freq.values())
    
    plt.figure(figsize=(10, 5))
    plt.hist(freq_dist, bins=30, color='lightgreen', edgecolor='black')
    plt.title(f'Word Frequency Distribution: {title}', fontsize=14)
    plt.xlabel('Frequency')
    plt.ylabel('Number of Words')
    plt.yscale('log') 
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"{title}_BoW_freq_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

def visualize_combined(texts, all_features, method='BoW', save_dir='figures'):
    """Visualiztion of all files."""
    combined_freq = Counter()
    for feature in all_features:
        combined_freq.update(feature)
    
    print(f"\nCombined {method} Visualizations (All Files):")
    plot_word_cloud(combined_freq, "All_Files", method, save_dir)
    plot_top_words(combined_freq, "All_Files", 30, method, save_dir)
    if method == 'BoW':
        plot_word_frequency_distribution(combined_freq, "All_Files", save_dir)

def visualize_files(file_paths):
    texts = load_text_files(file_paths)
    tfidf_matrix, tfidf_features = compute_tfidf(texts)
    bow_matrix, bow_features = compute_bow(texts)
    
    all_bow_features = []
    all_tfidf_features = []
    
    for i, (text, file_path) in enumerate(zip(texts, file_paths)):
        file_name = os.path.basename(file_path).replace('.txt', '')
        
        print(f"\n{'='*50}")
        print(f"Visualizations for: {file_name}")
        print(f"{'='*50}")
        
        # --- BoW Visualizations ---
        print("\nBag-of-Words Analysis:")
        bow_scores = bow_matrix[i].toarray().flatten()
        bow_freq = dict(zip(bow_features, bow_scores))
        all_bow_features.append(bow_freq)

        plot_top_words(bow_freq, file_name, 20, 'BoW')
        plot_word_cloud(bow_freq, file_name, 'BoW')
        plot_word_frequency_distribution(bow_freq, file_name)
        
        # --- TF-IDF Visualizations ---
        print("\nTF-IDF Analysis:")
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        tfidf_freq = dict(zip(tfidf_features, tfidf_scores))
        all_tfidf_features.append(tfidf_freq)

        plot_top_words(tfidf_freq, file_name, 20, 'TF-IDF')
        plot_word_cloud(tfidf_freq, file_name, 'TF-IDF')
    
    # --- Combined Visualizations ---
    print(f"\n{'='*50}")
    print("COMBINED VISUALIZATIONS FOR ALL FILES")
    print(f"{'='*50}")
    visualize_combined(texts, all_bow_features, 'BoW')
    visualize_combined(texts, all_tfidf_features, 'TF-IDF')

# --- Execution Entry Point ---
if __name__ == "__main__":
    input_files = [
        'preprocessed_chats/preprocessed_Clima_Jan_June.txt',
        'preprocessed_chats/preprocessed_Cocomelons.txt',
        'preprocessed_chats/preprocessed_marshmallows.txt',
        'preprocessed_chats/preprocessed_TMI_Central.txt'
    ]
    
    visualize_files(input_files)
