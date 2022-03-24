import os 

folder = ["1", "2", "3", "4", "5", "6", "7"]
for f in folder:
    print(f)
    r = os.popen("python .\mtb.py "+f)
    print(r)