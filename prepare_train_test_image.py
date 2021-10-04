
import pandas as pd
import os
# Shutil module offers a number of high-level operations on files and collections of files:
# copy, copymode, ...  
import shutil

path_train_csv = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train.csv'
path_test_csv = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\test.csv'
path_train_test_image = r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train_test'

train_csv = pd.read_csv(path_train_csv)
# making train directory
train_dir = os.mkdir(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train',mode=0o777, dir_fd=None)

needed_train_image_id = train_csv.loc[: , 'id'].values

filenames = []
for file in os.listdir(path_train_test_image):
    filenames.append(file) 
    
source=r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train_test'
destination=r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train'


# checking the matched image id  between train_set and train.csv into train file     
for image_id in needed_train_image_id :
    # shutil.copymode(source+'\\'+str(image_id)+'.png', destination)
    shutil.copy(source+'\\'+str(image_id)+'.png', destination)


test_csv = pd.read_csv(path_test_csv)
# making test directory
test_dir = os.mkdir(r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\test')           
needed_test_image_id = test_csv.loc[: , 'id'].values        

source=r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\train_test'
destination=r'D:\Work_Nubeslab\Invoice_Classification_Using ANN\test'    

# checking the matched image id  between train_set and test.csv into train file 
for image_id in needed_test_image_id :
    #shutil.copymode(source+'\\'+str(image_id)+'.png', destination)
    shutil.copy(source+'\\'+str(image_id)+'.png', destination)
    



