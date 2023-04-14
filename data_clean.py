import pandas as pd
import re
import csv

df= pd.read_csv('train-toneless_origin.csv')

pattern = re.compile("[^A-Za-z\s]")

id_list=df['id'].values.tolist()
text_list=df['text'].values.tolist()

for i in range(len(text_list)):
    temp=re.sub(pattern,'',text_list[i])
    if temp!=text_list[i]:
        print("clean id:{}".format(id_list[i]))
        text_list[i]=temp

with open('train.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile,delimiter=',')
    writer.writerow(["id", "text"])
    for i in range(len(id_list)):
        writer.writerow([id_list[i], text_list[i]])
