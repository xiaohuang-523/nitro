import tkinter as tk
from tkinter import filedialog
import os

# return without '\\' symbol at the end
def select_folder():
    folder_path = filedialog.askdirectory(initialdir="/", title="Select a Folder")
    return folder_path


def select_file(str = "Select a File"):
    file_path = filedialog.askopenfilename(initialdir="/", title=str)
    print('selected file '+ file_path)
    return file_path


# Returns the full path to the files inside a folder
def files_in_folder(folder_path, check_extension = 'N/A'):
    list = os.listdir(folder_path)
    container = []
    for item in list:
        if check_extension == 'N/A':
            container.append(folder_path + item)
        else:
            if item.endswith(check_extension):
                container.append(folder_path + item)
    return container


# Returns the full path to the folders inside a folder
def folders_in_folder(folder_path):
    list = os.listdir(folder_path)
    container = []
    for item in list:
        if os.path.isdir(folder_path + item):
            container.append(folder_path + item)
    return container
