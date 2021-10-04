#convert pdf to png
import os
from pdf2image import convert_from_path
import cv2
import pandas as pd

#file location
path = "/Users/feliciang/Downloads/predict_columbia"

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if(fullPath.endswith('.pdf')):
                allFiles.append(fullPath)
                
    return allFiles

def changeFileName(dirName):
    
    #load all file
    i = 1
   
    listOfFile = os.listdir(dirName)
    # Iterate over all the entries
    for file in listOfFile:
       
        if (file.endswith('.png')):
            filename = os.path.splitext(file)[0]
           
            #dst =filename + ".png"
            dst = str(i) + ".png"
            src = path + '/' +file 
            dst = path + '/' + dst 
            
            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 
            i += 1
    
def convertPDFToPNG(fileList):
    
    allTIFFFiles = list()
    #load all file
   
    for file in fileList: 
        filename = file
        pages = convert_from_path(filename)  
        total_pages = len(pages)
        
        for i in range(0, total_pages):  
            actualName = filename.split(".pdf")
            saveName= actualName[0] +'_'+ str(i+1)+'.png';
            pages[i].save(saveName, 'png')  
            allTIFFFiles.append(saveName)
     
    return allTIFFFiles   
       
def ImageProcessing(listOfTiff):
    
   root_dir = os.getcwd()
   file_list = ['train.csv', 'val.csv']
   image_source_dir = os.path.join(root_dir, 'data/images/')
   data_root = os.path.join(root_dir, 'data')
   for file in file_list:
    
       image_target_dir = os.path.join(data_root, file.split(".")[0])
    
       # read list of image files to process from file
       image_list = pd.read_csv(os.path.join(data_root, file), header=None)[0]
    
       print("Start preprocessing images")
       for image in image_list:
            # open image file
            if(image=='image_id'):
                continue
            img = cv2.imread(os.path.join(image_source_dir, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # perform transformations on image
            #b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
            #g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
            #r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
            
            # merge the transformed channels back to an image
            #transformed_image = cv2.merge((b, g, r))
            target_file = os.path.join(image_target_dir, image)
           
            print("Writing target file {}".format(target_file))
            cv2.imwrite(target_file, img)

def image_resize(imageList, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size

    # Iterate over all the entries
    for file in imageList:
    
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dim = None
        print(image)
        (h, w) = image.shape[:2]
    
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
    
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
    
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
    
        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)
        
        #store in to local folder
        actualName = file.split(".png")
        
        target_file = actualName[0] +'.png'
           
        print("Writing target file {}".format(target_file))
        cv2.imwrite(target_file, resized)
        # return the resized image
    return resized

if __name__ == "__main__":
    
    #loop all pdf file
    listOfFiles = getListOfFiles(path)  
    
    #convert pdf to every single png
    listOfTiff = convertPDFToPNG(listOfFiles)
    #image prcocessing for training
    ImageProcessing(listOfTiff)
    changeFileName(path)
    
    image = image_resize(listOfTiff, width=500, height = 500)

    
    
    
 