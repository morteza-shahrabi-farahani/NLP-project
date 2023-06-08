import os

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import csv

file_names = ['identity_hate_comments.csv', 'insult_comments.csv', 'obscene_comments.csv',
              'severe_toxic_comments.csv', 'threat_comments.csv', 'toxic_comments.csv',
              'usual_comments.csv']
read_path = "data/clean/"  # Set the path to your file
usual_comments = {}
unusual_comments = {}
toxic_comments = {}
severe_toxic_comments = {}
obscene_comments = {}
threat_comments = {}
insult_comments = {}
identity_hate_comments = {}
for name in file_names:
    path = read_path + name
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
          id, comment_text = row[0], row[1]
          if name == "usual_comments.csv":
              usual_comments[id] = comment_text
          # else:
          #   unusual_comments[id] = comment_text
          if name == "toxic_comments.csv":
              toxic_comments[id] = comment_text
          if name == "threat_comments.csv":
              threat_comments[id] = comment_text
          if name == "severe_toxic_comments.csv":
              severe_toxic_comments[id] = comment_text
          if name == "obscene_comments.csv":
              obscene_comments[id] = comment_text
          if name == "insult_comments.csv":
              insult_comments[id] = comment_text
          if name == "identity_hate_comments.csv":
              identity_hate_comments[id] = comment_text

comments = {"usual_comments": usual_comments,
            "toxic_comments":toxic_comments,
            "severe_toxic_comments": severe_toxic_comments,
            "obscene_comments": obscene_comments,
            "threat_comments": threat_comments,
            "insult_comments": insult_comments,
            "identity_hate_comments": identity_hate_comments}

os.makedirs("data/wordbroken", exist_ok=True)
os.makedirs("data/sentencebroken", exist_ok=True)

nltk.download('punkt')
folder_path = "data/"  # Set the path to your CSV file
tokenized_comments = {}
comment_sentences = {}
for key, value in comments.items():
  word_path = folder_path + "wordbroken/" + key + ".csv"
  with open(word_path, 'w', newline='', encoding='utf-8') as file:
      word_writer = csv.writer(file)
      word_writer.writerow(["id", "tokenized_comment_text"])  # Write the header row
      for id, comment_text in value.items():
        tokenized_comment = word_tokenize(comment_text)
        if key in tokenized_comments:
          for token in tokenized_comment:
            if token in tokenized_comments[key]:
              tokenized_comments[key][token] += 1
            else:
              tokenized_comments[key][token] = 1
        else:
          tokenized_comments[key] = {}
          for token in tokenized_comment:
            if token in tokenized_comments[key]:
              tokenized_comments[key][token] += 1
            else:
              tokenized_comments[key][token] = 1
        word_writer.writerow([id, tokenized_comment])  # Write each row of data

print("word tokenizing done")

for key, value in comments.items():
  sentence_path = folder_path + "sentencebroken/" + key + ".csv"
  with open(sentence_path, 'w', newline='', encoding='utf-8') as file:
      sentence_writer = csv.writer(file)
      sentence_writer.writerow(["id", "tokenized_comment_text"])  # Write the header row
      for id, comment_text in value.items():
        sentences = sent_tokenize(comment_text)  # A simple rule-based sentence splitter
        if key in comment_sentences:
          for sentence in sentences:
            if sentence in comment_sentences[key]:
              comment_sentences[key][sentence] += 1
            else:
              comment_sentences[key][sentence] = 1
        else:
          comment_sentences[key] = {}
          for sentence in sentences:
            if sentence in comment_sentences[key]:
              comment_sentences[key][sentence] += 1
            else:
              comment_sentences[key][sentence] = 1
        sentence_writer.writerow([id, sentences])  # Write each row of data

print("sentence tokenizing done")