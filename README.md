# TF-IDF Based Chatbot

## Project Overview

This project implements an information-retrieval-based chatbot using Natural Language Processing (NLP) techniques. The chatbot responds to user queries by identifying the most relevant answer from a predefined conversational dataset using TF-IDF vectorization and cosine similarity.

The project demonstrates practical understanding of text preprocessing, similarity-based retrieval, and evaluation of NLP systems.

---

## Methodology

1. User input is preprocessed through normalization steps such as lowercasing and punctuation removal.
2. Questions from the dataset and the user query are transformed into TF-IDF vectors.
3. Cosine similarity is computed between the user query vector and dataset vectors.
4. The response corresponding to the highest similarity score is returned as the chatbot output.

This approach is well suited for FAQ-style and domain-specific chatbot applications.

---

## Technologies and Libraries

* Python 3
* scikit-learn (TF-IDF Vectorizer, cosine similarity)
* pandas
* NumPy
* matplotlib

---

## Project Structure


project/
| chatbot.py                 # Main chatbot logic
│ preprocess.py              # Text preprocessing utilities
│ evaluate.py                # Evaluation and metrics generation
│ Conversation.csv           # Question–answer dataset
│ confusion_matrix_normalized.png
│ evaluation_report.txt
│ README.md
│ .gitignore



## How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/Lakshyyarlagadda/chatbot-tfidf.git
cd chatbot-tfidf
```

### Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib
```

### Execute the Chatbot

```bash
python chatbot.py
```

Enter a query in the terminal to receive the most relevant response from the dataset.

---

## Evaluation

The chatbot is evaluated using standard classification metrics. Performance analysis includes:

* Normalized confusion matrix
* Precision, recall, and F1-score

Evaluation artifacts are provided in:

* confusion_matrix_normalized.png
* evaluation_report.txt

These results help assess the effectiveness of similarity-based response retrieval.

---

## Key Learnings

* Practical application of TF-IDF for text representation
* Use of cosine similarity for semantic matching
* Evaluation of NLP systems using quantitative metrics
* Version control and project management using Git and GitHub

---

## Future Enhancements

* Incorporate word embedding techniques such as Word2Vec or GloVe
* Explore transformer-based models such as BERT
* Develop a web-based interface using Flask or Streamlit
* Introduce confidence thresholds for fallback responses

---

## Author

Lakshy Yarlagadda

GitHub: [https://github.com/Lakshyyarlagadda](https://github.com/Lakshyyarlagadda)
