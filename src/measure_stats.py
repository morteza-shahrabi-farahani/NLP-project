import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import csv
import matplotlib.pyplot as plt
import math

file_names = ['identity_hate_comments.csv', 'insult_comments.csv', 'obscene_comments.csv',
              'severe_toxic_comments.csv', 'threat_comments.csv', 'toxic_comments.csv',
              'usual_comments.csv']
clean_read_path = "data/clean/"  # Set the path to your file
clean_usual_comments = {}
clean_unusual_comments = {}
clean_toxic_comments = {}
clean_severe_toxic_comments = {}
clean_obscene_comments = {}
clean_threat_comments = {}
clean_insult_comments = {}
clean_identity_hate_comments = {}
for name in file_names:
    path = clean_read_path + name
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
          id, comment_text = row[0], row[1]
          if name == "usual_comments.csv":
              clean_usual_comments[id] = comment_text
          # else:
          #   unusual_comments[id] = comment_text
          if name == "toxic_comments.csv":
              clean_toxic_comments[id] = comment_text
          if name == "threat_comments.csv":
              clean_threat_comments[id] = comment_text
          if name == "severe_toxic_comments.csv":
              clean_severe_toxic_comments[id] = comment_text
          if name == "obscene_comments.csv":
              clean_obscene_comments[id] = comment_text
          if name == "insult_comments.csv":
              clean_insult_comments[id] = comment_text
          if name == "identity_hate_comments.csv":
              clean_identity_hate_comments[id] = comment_text

clean_comments = {"usual_comments": clean_usual_comments,
            "toxic_comments": clean_toxic_comments,
            "severe_toxic_comments": clean_severe_toxic_comments,
            "obscene_comments": clean_obscene_comments,
            "threat_comments": clean_threat_comments,
            "insult_comments": clean_insult_comments,
            "identity_hate_comments": clean_identity_hate_comments}

nltk.download('punkt')
folder_path = "data/"  # Set the path to your CSV file
clean_tokenized_comments = {}
clean_comment_sentences = {}
for key, value in clean_comments.items():
  for id, comment_text in value.items():
    clean_tokenized_comment = word_tokenize(comment_text)
    if key in clean_tokenized_comments:
      for token in clean_tokenized_comment:
        if token in clean_tokenized_comments[key]:
          clean_tokenized_comments[key][token] += 1
        else:
          clean_tokenized_comments[key][token] = 1
    else:
      clean_tokenized_comments[key] = {}
      for token in clean_tokenized_comment:
        if token in clean_tokenized_comments[key]:
          clean_tokenized_comments[key][token] += 1
        else:
          clean_tokenized_comments[key][token] = 1

for key, value in clean_comments.items():
  for id, comment_text in value.items():
    clean_sentences = sent_tokenize(comment_text)  # A simple rule-based sentence splitter
    if key in clean_comment_sentences:
      for clean_sentence in clean_sentences:
        if clean_sentence in clean_comment_sentences[key]:
          clean_comment_sentences[key][clean_sentence] += 1
        else:
          clean_comment_sentences[key][clean_sentence] = 1
    else:
      clean_comment_sentences[key] = {}
      for clean_sentence in clean_sentences:
        if clean_sentence in clean_comment_sentences[key]:
          clean_comment_sentences[key][clean_sentence] += 1
        else:
          clean_comment_sentences[key][clean_sentence] = 1

file_names = ['identity_hate_comments.csv', 'insult_comments.csv', 'obscene_comments.csv',
              'severe_toxic_comments.csv', 'threat_comments.csv', 'toxic_comments.csv',
              'usual_comments.csv']
raw_read_path = "data/raw/classified/"  # Set the path to your file
raw_usual_comments = {}
raw_unusual_comments = {}
raw_toxic_comments = {}
raw_severe_toxic_comments = {}
raw_obscene_comments = {}
raw_threat_comments = {}
raw_insult_comments = {}
raw_identity_hate_comments = {}
for name in file_names:
    path = raw_read_path + name
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
          id, comment_text = row[0], row[1]
          if name == "usual_comments.csv":
              raw_usual_comments[id] = comment_text
          # else:
          #   unusual_comments[id] = comment_text
          if name == "toxic_comments.csv":
              raw_toxic_comments[id] = comment_text
          if name == "threat_comments.csv":
              raw_threat_comments[id] = comment_text
          if name == "severe_toxic_comments.csv":
              raw_severe_toxic_comments[id] = comment_text
          if name == "obscene_comments.csv":
              raw_obscene_comments[id] = comment_text
          if name == "insult_comments.csv":
              raw_insult_comments[id] = comment_text
          if name == "identity_hate_comments.csv":
              raw_identity_hate_comments[id] = comment_text

raw_comments = {"usual_comments": raw_usual_comments,
            "toxic_comments": raw_toxic_comments,
            "severe_toxic_comments": raw_severe_toxic_comments,
            "obscene_comments": raw_obscene_comments,
            "threat_comments": raw_threat_comments,
            "insult_comments": raw_insult_comments,
            "identity_hate_comments": raw_identity_hate_comments}

folder_path = "data/"  # Set the path to your CSV file
raw_tokenized_comments = {}
raw_comment_sentences = {}
for key, value in raw_comments.items():
  for id, comment_text in value.items():
    raw_tokenized_comment = word_tokenize(comment_text)
    if key in raw_tokenized_comments:
      for token in raw_tokenized_comment:
        if token in raw_tokenized_comments[key]:
          raw_tokenized_comments[key][token] += 1
        else:
          raw_tokenized_comments[key][token] = 1
    else:
      raw_tokenized_comments[key] = {}
      for token in raw_tokenized_comment:
        if token in raw_tokenized_comments[key]:
          raw_tokenized_comments[key][token] += 1
        else:
          raw_tokenized_comments[key][token] = 1

for key, value in raw_comments.items():
  for id, comment_text in value.items():
    raw_sentences = sent_tokenize(comment_text)  # A simple rule-based sentence splitter
    if key in raw_comment_sentences:
      for raw_sentence in raw_sentences:
        if raw_sentence in raw_comment_sentences[key]:
          raw_comment_sentences[key][raw_sentence] += 1
        else:
          raw_comment_sentences[key][raw_sentence] = 1
    else:
      raw_comment_sentences[key] = {}
      for raw_sentence in raw_sentences:
        if raw_sentence in raw_comment_sentences[key]:
          raw_comment_sentences[key][raw_sentence] += 1
        else:
          raw_comment_sentences[key][raw_sentence] = 1

print("raw sentence tokenizing done")

print("raw word tokenizing done")

clean_words_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

clean_sentences_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

for key, value in clean_tokenized_comments.items():
  for word, word_count in clean_tokenized_comments[key].items():
    clean_words_count[key] += word_count

for key, value in clean_comment_sentences.items():
  for sentence, sentence_count in clean_comment_sentences[key].items():
    clean_sentences_count[key] += sentence_count

raw_words_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

raw_sentences_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

for key, value in raw_tokenized_comments.items():
  for word, word_count in raw_tokenized_comments[key].items():
    raw_words_count[key] += word_count

for key, value in raw_comment_sentences.items():
  for sentence, sentence_count in raw_comment_sentences[key].items():
    raw_sentences_count[key] += sentence_count

#draw table
# Comparison table for clean_word_count and raw_word_count
# Comparison of clean word count and raw word count
# Create a function to add labels to the bars
def add_bar_labels(ax):
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom')

# Comparison of raw word count
plt.figure(figsize=(10, 8))
ax1 = plt.gca()
ax1.bar(clean_words_count.keys(), clean_words_count.values(), label='Clean Word Count')
add_bar_labels(ax1)
plt.xlabel('Variable')
plt.ylabel('Count')
plt.title('Comparison of Raw Word Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stats/raw_word_count.png')
plt.close()

# Comparison of raw sentence count
plt.figure(figsize=(10, 8))
ax2 = plt.gca()
ax2.bar(raw_sentences_count.keys(), raw_sentences_count.values(), label='Raw Sentence Count')
add_bar_labels(ax2)
plt.xlabel('Variable')
plt.ylabel('Count')
plt.title('Comparison of Raw Sentence Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stats/raw_sentence_count.png')
plt.close()

# Comparison of clean sentence count
plt.figure(figsize=(10, 8))
ax3 = plt.gca()
ax3.bar(clean_sentences_count.keys(), clean_sentences_count.values(), label='Clean Sentence Count')
add_bar_labels(ax3)
plt.xlabel('Variable')
plt.ylabel('Count')
plt.title('Comparison of Clean Sentence Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stats/clean_sentence_count.png')
plt.close()

# Comparison of clean word count
plt.figure(figsize=(10, 8))
ax4 = plt.gca()
ax4.bar(clean_words_count.keys(), clean_words_count.values(), label='Clean Word Count')
add_bar_labels(ax4)
plt.xlabel('Variable')
plt.ylabel('Count')
plt.title('Comparison of Clean Word Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stats/clean_word_count.png')
plt.close()

# Make csv files
write_path = "stats/"
file_names = ['clean_word_count.csv', 'raw_word_count.csv', 'clean_sentence_count.csv', 'raw_sentence_count.csv']
variables = [clean_words_count, raw_words_count, clean_sentences_count, raw_sentences_count]

for name, variable in zip(file_names, variables):
    path = write_path + name
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Count"])  # Write the header row
        for category, count in variable.items():
            writer.writerow([category, count])  # Write each row of data

print('complete simple png and csv')
# next parts of report
unique_words_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

total_unique_words_count = 0

for key, value in clean_tokenized_comments.items():
  unique_words_count[key] = len(clean_tokenized_comments[key])
  total_unique_words_count += len(clean_tokenized_comments[key])

unique_common_words = {}
for outer_key, outer_value in clean_tokenized_comments.items():
  for word, word_count in clean_tokenized_comments[outer_key].items():
    for inner_key, inner_value in clean_tokenized_comments.items():
      if inner_key == outer_key:
        continue
      if word in clean_tokenized_comments[inner_key]:
        if word in unique_common_words:
          unique_common_words[word] += 1
        else:
          unique_common_words[word] = 0

unique_common_words_count = {"usual_comments": {},
            "toxic_comments": {},
            "severe_toxic_comments": {},
            "obscene_comments": {},
            "threat_comments": {},
            "insult_comments": {},
            "identity_hate_comments": {}}

for outer_key, outer_value in clean_tokenized_comments.items():
  for word, word_count in clean_tokenized_comments[outer_key].items():
    for inner_key, inner_value in clean_tokenized_comments.items():
      if inner_key == outer_key:
        continue
      if word in clean_tokenized_comments[inner_key]:
          if word in unique_common_words_count[outer_key]:
              unique_common_words_count[outer_key][word] += 1
          else:
              unique_common_words_count[outer_key][word] = 0

unique_common_words_total = {"usual_comments": len(unique_common_words_count["usual_comments"]),
            "toxic_comments": len(unique_common_words_count["toxic_comments"]),
            "severe_toxic_comments": len(unique_common_words_count["severe_toxic_comments"]),
            "obscene_comments": len(unique_common_words_count["obscene_comments"]),
            "threat_comments": len(unique_common_words_count["threat_comments"]),
            "insult_comments": len(unique_common_words_count["insult_comments"]),
            "identity_hate_comments": len(unique_common_words_count["identity_hate_comments"])}

plt.bar(unique_common_words_total.keys(), unique_common_words_total.values())
plt.xlabel('Comments')
plt.ylabel('Count')
plt.title('Unique Uncommon Words Count')
plt.xticks(rotation=75)
for i, count in enumerate(unique_common_words_total.values()):
    plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('stats/unique_common_words_total.png')
plt.close()

# CSV
csv_file = 'stats/unique_common_words_total.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Comment', 'Count'])
    writer.writerows(unique_common_words_total.items())

unique_uncommon_words_count = {"usual_comments": 0,
            "toxic_comments": 0,
            "severe_toxic_comments": 0,
            "obscene_comments": 0,
            "threat_comments": 0,
            "insult_comments": 0,
            "identity_hate_comments": 0}

for key, value in unique_uncommon_words_count.items():
    unique_uncommon_words_count[key] = unique_words_count[key] - unique_common_words_total[key]

plt.bar(unique_uncommon_words_count.keys(), unique_uncommon_words_count.values())
plt.xlabel('Comments')
plt.ylabel('Count')
plt.title('Unique Uncommon Words Count')
plt.xticks(rotation=75)
for i, count in enumerate(unique_uncommon_words_count.values()):
    plt.annotate(str(count), xy=(i, count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('stats/unique_uncommon_words_count.png')
plt.close()

# CSV
csv_file = 'stats/unique_uncommon_words_count.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Comment', 'Count'])
    writer.writerows(unique_uncommon_words_count.items())

unique_common_words_count = len(unique_common_words)
total_uncommon_unique_words = total_unique_words_count - unique_common_words_count

sorted_tokenized_words = {
    comment_type: dict(sorted(comment_dict.items(), key=lambda x: x[1], reverse=True))
    for comment_type, comment_dict in clean_tokenized_comments.items()
}

sorted_uncommon_tokenized_words = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for key, value in sorted_tokenized_words.items():
  for word, word_count in sorted_tokenized_words[key].items():
    if word not in unique_common_words:
      sorted_uncommon_tokenized_words[key][word] = word_count
      if len(sorted_uncommon_tokenized_words[key]) == 10:
        break

for key, values in sorted_uncommon_tokenized_words.items():
    # Sort the values dictionary by value in descending order
    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

    # Get the top ten entries
    top_ten = sorted_values[:10]
    top_ten = [list(entry) for entry in top_ten]
    for i in range(len(top_ten)):
        top_ten[i][0] = top_ten[i][0][:15]

    top_ten = [tuple(entry) for entry in top_ten]
    # Create lists for keys and values
    keys = [entry[0] for entry in top_ten]
    frequency = [entry[1] for entry in top_ten]

    # Plot the data
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(keys, frequency)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Top Ten Tokens - {key}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Save the plot as a PNG file
    plt.savefig(f'stats/top_ten_tokens_{key}.png')
    plt.close()

    # Save the top ten entries in a CSV file
    csv_file = f'stats/top_ten_tokens_{key}.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Frequency'])  # Write the header row
        writer.writerows(top_ten)  # Write the data rows

print('complete part before RNE')
#RNE scores
RNE_words_score = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for word, word_count in unique_common_words.items():
  for RNE_class, RNE_dict in RNE_words_score.items():
    if word in clean_tokenized_comments[RNE_class]:
      word_in_class_count = clean_tokenized_comments[RNE_class][word]
      RNE_score = (word_in_class_count / unique_words_count[RNE_class]) / (unique_common_words[word] / unique_common_words_count)
      RNE_words_score[RNE_class][word] = RNE_score

sorted_RNE_scores = {
    comment_type: dict(sorted(comment_dict.items(), key=lambda x: x[1], reverse=True))
    for comment_type, comment_dict in RNE_words_score.items()
}

visualized_RNE_words = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for key, value in sorted_RNE_scores.items():
  for word, word_count in sorted_RNE_scores[key].items():
    if word == '.' or word == ',' or word == '!' or word == "''" or word == "``":
      continue
    else:
      visualized_RNE_words[key][word] = word_count
    if len(visualized_RNE_words[key]) == 10:
      break

for key, values in visualized_RNE_words.items():
    # Sort the values dictionary by value in descending order
    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

    # Get the top ten entries
    top_ten = sorted_values[:10]
    top_ten = [list(entry) for entry in top_ten]

    # Create lists for keys and values
    keys = [entry[0] for entry in top_ten]
    frequency = [entry[1]for entry in top_ten]  # Format the value with four decimal places

    # Plot the data
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(keys, frequency)
    plt.xlabel('Tokens')
    plt.ylabel('Value')
    plt.title(f'Top Ten RNE - {key}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Save the plot as a PNG file
    plt.savefig(f'stats/top_ten_RNE_{key}.png')
    plt.close()

    for i in range(len(top_ten)):
        top_ten[i][1] = f"{top_ten[i][1]:.4f}"

    top_ten = [tuple(entry) for entry in top_ten]
    # Save the top ten entries in a CSV file
    csv_file = f'stats/top_ten_RNE_{key}.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Value'])  # Write the header row
        writer.writerows(top_ten)  # Write the data rows

print('complete RNE')

TF_IDF_words_score = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for word, word_count in unique_common_words.items():
  for TF_IDF_class, TF_IDF_dict in TF_IDF_words_score.items():
    if word in clean_tokenized_comments[TF_IDF_class]:
      word_in_class_count = clean_tokenized_comments[TF_IDF_class][word]
      TF_score = (word_in_class_count / unique_words_count[TF_IDF_class])
      classes = TF_IDF_words_score.keys()
      containing_documents = 0
      for inner_class in classes:
        if word in clean_tokenized_comments[inner_class]:
          containing_documents += 1
      IDF_score = math.log(len(classes) / containing_documents)
      TF_IDF_score = TF_score * IDF_score
      TF_IDF_words_score[TF_IDF_class][word] = TF_IDF_score

sorted_TF_IDF_scores = {
    comment_type: dict(sorted(comment_dict.items(), key=lambda x: x[1], reverse=True))
    for comment_type, comment_dict in TF_IDF_words_score.items()
}

visualized_TF_IDF_words = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for key, value in sorted_TF_IDF_scores.items():
  for word, word_count in sorted_TF_IDF_scores[key].items():
    if word == '.' or word == ',' or word == '!' or word == "''" or word == "``":
      continue
    else:
      visualized_TF_IDF_words[key][word] = word_count
    if len(visualized_TF_IDF_words[key]) == 10:
      break


for key, values in visualized_TF_IDF_words.items():
    # Sort the values dictionary by value in descending order
    sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

    # Get the top ten entries
    top_ten = sorted_values[:10]
    top_ten = [list(entry) for entry in top_ten]
    for i in range(len(top_ten)):
        top_ten[i][0] = top_ten[i][0][:23]

    top_ten = [tuple(entry) for entry in top_ten]

    # Create lists for keys and values
    keys = [entry[0] for entry in top_ten]
    frequency = [entry[1] for entry in top_ten]

    # Plot the data
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(keys, frequency)
    plt.xlabel('Tokens')
    plt.ylabel('Value')
    plt.title(f'Top Ten TF-IDF - {key}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Save the plot as a PNG file
    plt.savefig(f'stats/top_ten_TF-IDF_{key}.png')
    plt.close()

    top_ten = [list(entry) for entry in top_ten]

    for i in range(len(top_ten)):
        top_ten[i][1] = f"{top_ten[i][1]:.8f}"

    top_ten = [tuple(entry) for entry in top_ten]
    # Save the top ten entries in a CSV file
    csv_file = f'stats/top_ten_TF-IDF_{key}.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Value'])  # Write the header row
        writer.writerows(top_ten)  # Write the data rows

print('complete TF-IDF')

sorted_histogram_tokenized_words = {
    "usual_comments": {},
    "toxic_comments": {},
    "severe_toxic_comments": {},
    "obscene_comments": {},
    "threat_comments": {},
    "insult_comments": {},
    "identity_hate_comments": {}
}

for key, value in sorted_tokenized_words.items():
  for word, word_count in sorted_tokenized_words[key].items():
    if word == '.' or word == ',' or word == '!' or word == "''" or word == "``":
      continue
    else:
      sorted_histogram_tokenized_words[key][word] = word_count
    if len(sorted_histogram_tokenized_words[key]) == 10:
      break

for category, word_freq in sorted_histogram_tokenized_words.items():
    # Get the top N words and their frequencies for the histogram
    N = 10
    top_words = list(word_freq.keys())[:N]
    top_freqs = list(word_freq.values())[:N]

    # Create the histogram
    plt.figure()
    plt.bar(top_words, top_freqs)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top {} Words Frequency - {}'.format(N, category))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'stats/top_ten_histogram_{category}.png')
    plt.close()

    csv_file = f'stats/top_ten_histogram_{category}.csv'
    top_ten = list(zip(top_words, top_freqs))

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Value'])  # Write the header row
        writer.writerows(top_ten)  # Write the data rows
# Display all the histograms

print("completed finally")

