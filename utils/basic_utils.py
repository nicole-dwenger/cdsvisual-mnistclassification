#!/usr/bin/python

"""
Utility functions for basic steps, e.g. creating directories or filenames
"""

import os
import sys



def create_dir(dir_name):
    """
    Creating directory in the current directory, if it does not exist already
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
 
        
def unique_path(filepath):
    """
    Creating unique filepath/filename by incrementing filename, if the filename exists already 
    """ 
    # If the path does not exist, keep the original filepath
    if not os.path.exists(filepath):
        return filepath
    
    # Otherwise, split the path, and append a number that does not exist already, return the new path
    else:
        i = 1
        path, ext = os.path.splitext(filepath)
        new_path = "{}_{}{}".format(path, i, ext)
        
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path 
    
    
def get_filepaths(root_directory):
    """
    Creating list of filepaths for all files with extensitions in the directory and subdirectories
    - Directory: Root directory
    """
    # Define extensions
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    # Empty list for file paths
    filepaths = []
    # Initialise counter
    counter = 1
    
    # Loop through directory and append filepaths
    for root, directories, filenames in os.walk(root_directory):
        for filename in filenames:
            # If the file has one of the extensions
            if any(ext in filename for ext in extensions):
                filepaths.append(os.path.join(root_directory, filename))
                # Increment counter
                counter += 1
                
    return file_list

                      
if __name__=="__main__":
    pass


    