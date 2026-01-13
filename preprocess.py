import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_and_preprocess_data(file_path):
    print(" Loading data...")
    df = pd.read_csv(file_path)
    print(" Columns found:", df.columns.tolist())
    questions_raw = df['question'].astype(str).tolist()
    answers = df['answer'].astype(str).tolist()
    questions = [preprocess_text(q) for q in questions_raw]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    with open('questions.pkl', 'wb') as q_file:
        pickle.dump(questions, q_file)
    with open('answers.pkl', 'wb') as a_file:
        pickle.dump(answers, a_file)
    with open('vectorizer.pkl', 'wb') as v_file:
        pickle.dump(vectorizer, v_file)
    print(" Preprocessing complete!")
    return questions, answers

if __name__ == '__main__':
    load_and_preprocess_data('Conversation.csv')
