import zipfile
import os
import kagglehub
import pandas as pd
import random
import cv2
import functions_and_classes as f
from collections import defaultdict


# Download specific version of data
path = kagglehub.dataset_download("hgunraj/covidxct/versions/2")

# SPECIFIC TO USER #
print("Path to dataset files:", path)
zip_path = "/Users/scottdavidson/Downloads/archive.zip"
docs_path = 'train_test_docs'

train_test_docs = ['train_COVIDx_CT-2A','val_COVIDx_CT-2A', 'test_COVIDx_CT-2A']

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get the list of all files in the ZIP
    all_files = zip_ref.namelist()

    # Extract train, val, test files:
    for file in all_files:
        if any(sub in file for sub in train_test_docs):
            zip_ref.extract(member=file, path=docs_path)


### Convert each file into a data frame ###
def convert_to_df(doc):
    df = pd.read_csv(docs_path + '/' + doc + '.txt', sep=' ')
    # Adjust columns & index
    cols = df.columns
    df.loc[-1] = cols
    df.index = df.index + 1
    df = df.sort_index()
    df.columns = ['File','COVID_19_Status','2','3','4','5'] # COVID_19_Status: 0= No covid, 1= pneumonia, 2=Covid
    return(df)


docs_df = [convert_to_df(doc) for doc in train_test_docs]
train = docs_df[0].copy()
val = docs_df[1].copy()
test = docs_df[2].copy()


### Get random samples from each grouping of files ###
def extract_files(df, df_type, proportion):
    # Separate by covid vs non-covid & extract randomly sampled files from the zip
    covid_files = df.loc[df['COVID_19_Status'] == 2, 'File'].to_list() # 2 = COVID
    non_covid_files = df.loc[df['COVID_19_Status'] == 0, 'File'].to_list() # 0 = NO COVID
    print(df_type, len(covid_files), len(non_covid_files))

    sample_covid = random.sample(covid_files, int(len(covid_files) * proportion))
    sample_non_covid = random.sample(non_covid_files, int(len(non_covid_files) * proportion))

    return(sample_covid, sample_non_covid)


train_covid_files, train_non_covid_files = extract_files(train, 'Train', 1) # Determined by looking at data amount & not getting too much
val_covid_files, val_non_covid_files = extract_files(val, 'Val', 1)
test_covid_files, test_non_covid_files = extract_files(test, 'Test', 1)

### Output each file grouping to a folder in the directory ###
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
# Get the list of all files in the ZIP
    all_files = zip_ref.namelist()

    for file in all_files:
        # Train
        if any(sub in file for sub in train_covid_files):
            zip_ref.extract(member=file, path='Train/YES_COVID')
        elif any(sub in file for sub in train_non_covid_files):
            zip_ref.extract(member=file, path='Train/NOT_COVID')

        # Val
        elif any(sub in file for sub in val_covid_files):
            zip_ref.extract(member=file, path='Val/YES_COVID')
        elif any(sub in file for sub in val_non_covid_files):
            zip_ref.extract(member=file, path='Val/NOT_COVID')

        # Test
        elif any(sub in file for sub in test_covid_files):
            zip_ref.extract(member=file, path='Test/YES_COVID')
        elif any(sub in file for sub in test_non_covid_files):
            zip_ref.extract(member=file, path='Test/NOT_COVID')