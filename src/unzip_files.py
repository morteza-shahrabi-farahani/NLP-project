import zipfile

zip_file_path1 = './data/raw/initial-files/same_submission.zip'  # Path to the zip file
zip_file_path2 = './data/raw/initial-files/test_labels.zip'  # Path to the zip file
zip_file_path3 = './data/raw/initial-files/train.zip'  # Path to the zip file
zip_file_path4 = './data/raw/initial-files/test.zip'  # Path to the zip file
extract_to_directory = './data/raw/initial-files'  # Directory to extract the contents to
# Open the zip file
with zipfile.ZipFile(zip_file_path1, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_to_directory)

with zipfile.ZipFile(zip_file_path2, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_to_directory)

with zipfile.ZipFile(zip_file_path3, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_to_directory)

with zipfile.ZipFile(zip_file_path4, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_to_directory)

print('File extracted successfully.')