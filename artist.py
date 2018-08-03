import os
import sys
import shutil
import time

name = sys.argv[1]

material_path = "material" + os.sep + name
images_path = "image_data" + os.sep
remove_path = images_path + "*"
print("remove " + remove_path + " after 5 seconds.")
time.sleep(5)

##############################################################
os.sysmte("rm -rf " + remove_path)

os.system("python3 collector.py " + name)
os.system("mv " + images_path + "*")
os.system("python3 preprocessor.py")
os.system("python3 gan.py")
os.system("python3 chrome_theme.py " + name)

