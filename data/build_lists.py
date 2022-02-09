# Split caetagories according to the provided list
import csv
import os, os.path
import glob

# This generates list for Sp2 Dataset
s1=['cricket','soccer','hockey','snooker']
s2=['football','icehockey','baseball','tennis']
s3=['rugby','badminton','tabletennis']
s4=['handball','volleyball','basketball']

s1_w = open("s1_1.csv", "w")
s2_w = open("s2_1.csv", "w")
s3_w = open("s3_1.csv", "w")
s4_w = open("s4_1.csv", "w")

# reading and splitiing the CSV File
with open('Splits.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        cat=row[1]
        
        # Count the images against each list
        splits=row[2].split('.')
        splits2=splits[0].split('_')
        pth=splits2[2]
        
        fol=splits[0].split('/')
        
        
        
        num_files=len(glob.glob(os.path.join(os.path.join('SP2',row[1],pth,fol[1],'*.jpg'))))
        #print(num_files)
        line=row[0],',',row[1],',',row[2],',',row[3],',',row[4],',',row[5],',',str(num_files),'\n'
        line=''.join(line)
        if cat in s1:
            s1_w.write(line)
        if cat in s2:
            s2_w.write(line)
        if cat in s3:
            s3_w.write(line)            
        if cat in s4:
            s4_w.write(line)            
                        





