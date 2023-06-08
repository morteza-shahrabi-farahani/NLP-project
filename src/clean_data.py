import csv
import os


read_path = "data/raw/initial-files/train.csv"  # Set the path to your file
usual_comments = {}
unusual_comments = {}
toxic_comments = {}
severe_toxic_comments = {}
obscene_comments = {}
threat_comments = {}
insult_comments = {}
identity_hate_comments = {}
with open(read_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
      (id, comment_text, toxic, severe_toxic, obscene, threat, insult,
      identity_hate) = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
      if(toxic == '0' and severe_toxic == '0' and obscene == '0' and threat == '0'
      and insult == '0' and identity_hate == '0'):
        usual_comments[id] = comment_text
      # else:
      #   unusual_comments[id] = comment_text
      if toxic == '1':
        toxic_comments[id] = comment_text
      if severe_toxic == '1':
        severe_toxic_comments[id] = comment_text
      if obscene == '1':
        obscene_comments[id] = comment_text
      if threat == '1':
        threat_comments[id] = comment_text
      if insult == '1':
        insult_comments[id] = comment_text
      if identity_hate == '1':
        identity_hate_comments[id] = comment_text

comments = {"usual_comments": usual_comments,
            "toxic_comments":toxic_comments,
            "severe_toxic_comments": severe_toxic_comments,
            "obscene_comments": obscene_comments,
            "threat_comments": threat_comments,
            "insult_comments": insult_comments,
            "identity_hate_comments": identity_hate_comments}

write_path = "data/clean/"  # Set the path to your CSV file

for key, value in comments.items():
    path = write_path + key + ".csv"
    # Create the directory if it doesn't exist
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Create the file if it doesn't exist
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8'):
            pass  # Create an empty file

    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "comment_text"])  # Write the header row
        for id, comment_text in value.items():
            writer.writerow([id, comment_text.lower()])  # Convert comment_text to lowercase before writing


print("cleaning done")