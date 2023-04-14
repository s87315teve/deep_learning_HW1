import time
import os
import datetime 
import subprocess
import csv


origin_folder="train_origin"
new_folder="train"
#train_origin
folder_exist=os.path.isdir("train")
if not folder_exist:
    os.mkdir("train")
file_list=[f for f in os.listdir(origin_folder)]
for filename in file_list:
    #print("{}/{} -r 16000 -e signed-integer -b 16 {}/{}".format(origin_folder, filename, new_folder, filename))
    subprocess.getstatusoutput("sox {}/{} -r 16000 -e signed-integer -b 16 {}/{}".format(origin_folder, filename, new_folder, filename))

print("train folder finished")


origin_folder="test_origin"
new_folder="test"
#train_origin
folder_exist=os.path.isdir("test")
if not folder_exist:
    os.mkdir("test")
file_list=[f for f in os.listdir(origin_folder)]
for filename in file_list:
    subprocess.getstatusoutput("sox {}/{} -r 16000 -e signed-integer -b 16 {}/{}".format(origin_folder, filename, new_folder, filename))
print("test folder finished")
