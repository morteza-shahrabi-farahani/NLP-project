import os
import zipfile
import requests

url = 'https://imgurl.ir/uploads/w529972_same_submission.csv.zip'
filename = 'same_submission.zip'  # The name you want to give to the downloaded file
directory = './data/raw/initial-files'  # The directory path where the file should be saved

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
# Request for getting zip file
response = requests.get(url)

if response.status_code == 200:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as file:
        file.write(response.content)
    print(f"File '{filename}' downloaded successfully.")
else:
    print(f"Failed to download the file. Error code: {response.status_code}")

url2 = 'https://imgurl.ir/uploads/c796002_test_labels.csv.zip'
filename2 = 'test_labels.zip'  # The name you want to give to the downloaded file
directory = './data/raw/initial-files'  # The directory path where the file should be saved

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
response = requests.get(url2)

if response.status_code == 200:
    filepath = os.path.join(directory, filename2)
    with open(filepath, 'wb') as file:
        file.write(response.content)
    print(f"File '{filename2}' downloaded successfully.")
else:
    print(f"Failed to download the file. Error code: {response.status_code}")

url3 = 'https://imgurl.ir/uploads/j67995_train.csv.zip'
filename3 = 'train.zip'  # The name you want to give to the downloaded file
directory = './data/raw/initial-files'  # The directory path where the file should be saved

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
response = requests.get(url3)

if response.status_code == 200:
    filepath = os.path.join(directory, filename3)
    with open(filepath, 'wb') as file:
        file.write(response.content)
    print(f"File '{filename3}' downloaded successfully.")
else:
    print(f"Failed to download the file. Error code: {response.status_code}")

url4 = 'https://imgurl.ir/uploads/z30706_test.csv.zip'
filename4 = 'test.zip'  # The name you want to give to the downloaded file
directory = './data/raw/initial-files'  # The directory path where the file should be saved

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)
response = requests.get(url4)

if response.status_code == 200:
    filepath = os.path.join(directory, filename4)
    with open(filepath, 'wb') as file:
        file.write(response.content)
    print(f"File '{filename4}' downloaded successfully.")
else:
    print(f"Failed to download the file. Error code: {response.status_code}")


# Download urls
#https://imgurl.ir/uploads/w529972_same_submission.csv.zip
#https://imgurl.ir/uploads/c796002_test_labels.csv.zip
#https://imgurl.ir/uploads/j67995_train.csv.zip
#https://imgurl.ir/uploads/z30706_test.csv.zip