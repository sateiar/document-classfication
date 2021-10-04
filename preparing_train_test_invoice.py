
path_related = r"D:\Work_Nubeslab\RelateInvoice.zip"
path_nonrelated = r"D:\Work_Nubeslab\non_RelatedInvoice.zip"

# unzip invoice files
import zipfile
with zipfile.ZipFile(path_related) as zip_ref:
    zip_ref.extractall('D:\Work_Nubeslab\Related_Invoive')
  
with zipfile.ZipFile(path_nonrelated) as zip_ref:
    zip_ref.extractall(r'D:\Work_Nubeslab\noRelated_Invoice')

# Changing filenames
import os    
i = 1
    listOfFile = os.listdir(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\Relevant_invoice')
    # Iterate over all the entries
    for file in listOfFile:
        
        dst = str(i) + ".png"
        src = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\Relevant_invoice' + "\\" + file 
        dst = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\Relevant_invoice'+ "\\" + dst 
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
        
    i = 4051
    listOfFile = os.listdir(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\non_Relevant_invoice')
    # Iterate over all the entries
    for file in listOfFile:
        
        dst = str(i) + ".png"
        src = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\non_Relevant_invoice' + "\\" + file 
        dst = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\non_Relevant_invoice'+ "\\" + dst 
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
        
import pandas as pd
dataset = pd.read_csv(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size = 0.20, random_state = 0)

# export traing set to a csv file
train.to_csv(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\train.csv',
              index = None, header = True)

# export test set to a csv file
test.to_csv(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\dataset_new_merge\test.csv',
            index = None, header = True)


















