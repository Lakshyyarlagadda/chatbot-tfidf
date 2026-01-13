from chatbot import chatbot_response
from preprocess import load_and_preprocess_data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def evaluate_model():
    questions_raw, answers = load_and_preprocess_data('Conversation.csv')
    y_true = answers
    y_pred = []
    print(" Running chatbot predictions...")
    for question in questions_raw:
        predicted_answer = chatbot_response(question)
        y_pred.append(predicted_answer)
    print("\n === Evaluation Metrics ===")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f" Accuracy     : {accuracy:.4f}")
    print(f" Precision    : {precision:.4f}")
    print(f" Recall       : {recall:.4f}")
    print(f" F1 Score     : {f1:.4f}")
    print("\n Classification Report:")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    print("\n Normalized Confusion Matrix:")
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=False, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png")
    plt.show()
    with open("evaluation_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write("=== Evaluation Metrics ===\n")
        report_file.write(f"Accuracy     : {accuracy:.4f}\n")
        report_file.write(f"Precision    : {precision:.4f}\n")
        report_file.write(f"Recall       : {recall:.4f}\n")
        report_file.write(f"F1 Score     : {f1:.4f}\n\n")
        report_file.write("=== Classification Report ===\n")
        report_file.write(report)
    print("\n Saved:")
    print(" - Confusion matrix: confusion_matrix_normalized.png")
    print(" - Evaluation report: evaluation_report.txt")

if __name__ == '__main__':
    evaluate_model()
