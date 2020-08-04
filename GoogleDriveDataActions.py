import os 
import subprocess

class GoogleDriveDataActions:
  def __init__(self, folder_name):
    self.folder_name = folder_name
  
  def check_else_mount_drive():
    if os.path.isdir("./drive") is True:
        from google.colab import drive
        drive.mount('/content/drive')
        print("The google drive mounted and is ready to accessed")
    else:
      print("The google drive is already mounted and can receive actions")
    return self


  # check if the warehouse directory exists else create it
  @staticmethod
  def check_warehouse_existence_else_create(folder_name):
        path_to_datasets = "/content/drive/My Drive/"+folder_name
        if os.path.isdir(path_to_datasets) is False:
          status_mkdir = subprocess.run(["mkdir", path_to_datasets])
          if status_mkdir.returncode == 1: print("An empty <data_for_dataScience> folder created to store the datasets")  
          return
        else:
          print("A folder warehouse <data_for_dataScience> in My Drive and has the datasets:")
          list_files = os.listdir(path_to_datasets)
          return list_files 


name_warehouse = "data_for_dataScience"
datascience_warehouse = GoogleDriveDataActions(name_warehouse)
files_of_warehouse = check_warehouse_existence_else_create.__get__(datascience_warehouse)(name_warehouse) # use of descriptor protocol
print(files_of_folder)
