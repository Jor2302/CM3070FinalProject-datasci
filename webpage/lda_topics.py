import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os

def train_lda_model(data_path="data/udemy_course_data.csv", n_topics=5):
    df = pd.read_csv(data_path)
    df['text'] = df['course_title'].fillna('') + " " + df['subject'].fillna('')
    
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(df['text'])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [terms[i] for i in topic.argsort()[-5:]]
        topics.append((f"Topic {idx+1}", top_words))
    return topics

def plot_topic_summary(topics, output_path="static/lda_topics.png"):
    os.makedirs("static", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [t[0] for t in topics]
    word_labels = [" | ".join(t[1]) for t in topics]
    ax.barh(labels, [len(t[1]) for t in topics], color="#3498db")
    for i, words in enumerate(word_labels):
        ax.text(len(topics[i][1]) + 0.1, i, words, va='center', fontsize=9)
    ax.set_title("Top 5 Words per LDA Topic")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
