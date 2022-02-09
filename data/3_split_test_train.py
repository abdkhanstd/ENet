import os
import csv
import glob
import os.path
import random

#############
Training_percentage=70


f = open('Splits.csv','w')
csv_data=[]
# Use appropriate random seeds
random.seed(30)


Training_percentage=Training_percentage/100
Test_percentage=1-Training_percentage

print(' Using ',Training_percentage,' for training and ',Test_percentage,' for testing')

# Get folder names in each caetagory
caetagory = glob.glob(os.path.join('SP2','*'))  


for cat_name in caetagory:
    # get groups in the folders
    gropus = glob.glob(os.path.join(cat_name,'*')) 
    num_groups=len(gropus)
    
    Train_groups=random.sample(range(1, num_groups), round(num_groups*Training_percentage))
    Test_groups=[]
    # Now creating the test groups
    for i in range(1,num_groups+1):
        if i not in Train_groups:
            Test_groups.append(i)
   
    
    # Now update the datafile.csv accordingly    
    with open('data_file.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            # Split the caetagory name and group number
            
            cat_name___=cat_name.split('/')
            cat_name__=cat_name___[1]
            
            #get the number of images for that file
            
            caetagory=row[0]
            if cat_name__==caetagory:            
                tmp=row[1].split('/')
                tmp=tmp[1].split('.')
                tmp=tmp[0].split('_')
                group=tmp[2]
                
                file=row[1]
                view=row[2]
                action=row[3]
                situation=row[4]

                # get actual foldername
                tmp=file.split('/')
                tmp=tmp[1].split('.')
                folder_name=tmp[0]
  
                num_frames = len(glob.glob(os.path.join(cat_name,group,folder_name,'*.jpg')))
                # Now update csv file accordingly
                if int(group) in Train_groups:
                    to_write='Train'+','+caetagory+','+file+','+view+','+action+','+situation+','+str(num_frames)+'\n'
                else:
                    to_write='Test'+','+caetagory+','+file+','+view+','+action+','+situation+','+str(num_frames)+'\n'
                    
                f.write(to_write)
                


            
            
            
            
            
            









        

    