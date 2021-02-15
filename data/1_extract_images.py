

import glob
import os
import os.path
from subprocess import call


def extract_files():
    # My program to extract files using ffmpg
    
    
    # make sure to add relative path
    path_to_videos='SP2'
    
    classes = glob.glob(os.path.join(path_to_videos,'*')) 
    print(classes)
    # Read folders
    for folder in classes:
        class_folders = glob.glob(os.path.join(folder, '*'))

        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.avi'))
            for video_file in class_files:
                src=video_file
                #Exploding file path to create a folder
                
                new_folder=src.split(".")
                dest_folder=new_folder[0]
                dest=os.path.join(dest_folder, '%05d.jpg')
                print(dest)
                #Create a folder
                if not os.path.exists(dest_folder):
                    os.mkdir(dest_folder) 
                call(["ffmpeg", "-threads","16","-i", src, dest])
                call(["rm", src])
            

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()