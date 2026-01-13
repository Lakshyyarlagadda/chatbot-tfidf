import pickle
import nltk
import string
from sklearn.metrics.pairwise import cosine_similarity
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

with open('questions.pkl', 'rb') as q_file:
    questions = pickle.load(q_file)

with open('answers.pkl', 'rb') as a_file:
    answers = pickle.load(a_file)

with open('vectorizer.pkl', 'rb') as v_file:
    vectorizer = pickle.load(v_file)

question_vectors = vectorizer.transform(questions)

def chatbot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarity_scores.argmax()
    best_score = similarity_scores[0, best_match_idx]
    if best_score >= 0.3:
        return answers[best_match_idx]
    else:
        return "Sorry, I don't understand that yet."

if __name__ == '__main__':
    print(" Chatbot is running! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print(" Goodbye!")
            break
        response = chatbot_response(user_input)
        print("Bot:", response)
