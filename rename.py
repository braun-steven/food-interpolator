import os
import sys
import shutil
import re

d = sys.argv[1]
for f in os.listdir(d):
    if "sample" in f:
        num = f.split("samples_")[1].split(".png")[0]
        new_name = f"samples_{num.zfill(4)}.png"
        print(new_name)
        shutil.move(os.path.join(d, f), os.path.join(d, new_name))
