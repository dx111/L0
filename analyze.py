from glob import glob
import pandas as pd
import csv
import io
from collections import OrderedDict

class CustomDialect(csv.excel):
    delimiter = ","


if __name__=="__main__":
    fieldnames =["acc"]+["val_acc"]+["type"]+["sigma"]+["lambda"]
    for i in range(4):
        fieldnames+=["n%s"%str(i+1)]
    for i in range(4):
        fieldnames+=["sp%s"%str(i+1)]
    csv_file = io.open("result.csv","w",newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect=CustomDialect)
    writer.writeheader()
    all_file=glob("log/*/val_10_fold.csv")
    for f in all_file:
        print(f)
        data=pd.read_csv(f)
        row=OrderedDict()
        row["acc"]=data["acc"].mean()
        row["val_acc"]=data["val_acc"].mean()
        row["type"]=data["type"][0]
        row["sigma"]=data["sigma"][0]
        row["lambda"]=data["reg_1"][0]
        row["n1"]=data["n1"].mean()
        row["n2"]=data["n2"].mean()
        row["n3"]=data["n3"].mean()
        row["n4"]=data["n4"].mean()
        row["sp1"]=data["sp1"].mean()
        row["sp2"]=data["sp2"].mean()
        row["sp3"]=data["sp3"].mean()
        row["sp4"]=data["sp4"].mean()
        writer.writerow(row)
        csv_file.flush()

