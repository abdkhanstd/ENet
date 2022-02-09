import os
import csv
import glob
new=[]
f = open('data_file.csv','w')
with open('V2.csv', 'r') as fin:
    reader = csv.reader(fin)
    for row in reader:
    
    
        #1 Get the class name
        tmp=row[0].split("/")
        sport_type=tmp[0]
        
        file=row[0]
        if not row[1]=='':
            view=sport_type+'_'+row[1]
        else:
            view=row[1] 


        if not row[2]=='':          
            action=sport_type+'_'+row[2]
        else:
            action=row[2]    

        if not row[3]=='':              
            situation=sport_type+'_'+row[3]
        else:
            situation=row[3]        
        
        
        to_write=sport_type+','+file+','+view+','+action+','+situation+'\n'
        
        print(to_write)
        f.write(to_write)

        
print('Done')        
        