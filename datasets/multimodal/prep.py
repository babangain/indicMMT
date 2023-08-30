import pandas as pd
import os
from PIL import Image
import secrets
VISUAL_GENOME_DIR='malayalam-visual-genome-10/'
CURRENT_DIR = os.getcwd()
CSV_FILENAME = VISUAL_GENOME_DIR + "malayalam-visual-genome-train.txt"
CSV_DIR = VISUAL_GENOME_DIR+CSV_FILENAME
IMG_DIR = VISUAL_GENOME_DIR + "malayalam-visual-genome-train.images/"
PROCESSED_IMG_DIR = IMG_DIR[:-1] + ".processed/"
os.mkdir(PROCESSED_IMG_DIR)

data = pd.read_csv(CSV_FILENAME, sep="\t", names=["imgId", "x0","y0", "width", "height", "en", "hi"])
print(data.head())
l = len(data)

hashes = []
for i in range(l):
	hsh = secrets.token_hex(nbytes=16)
	print(i)
	#imgID,x0,y0,width,height,en,hi = data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], data.iloc[i][3], data.iloc[i][4], data.iloc[i][5], data.iloc[i][6]
	imgID, x0, y0, width, height, en, hi = data.iloc[i,:].tolist()
	im = Image.open(IMG_DIR+str(imgID)+".jpg")
	im.save(PROCESSED_IMG_DIR + hsh+"_full.jpg")
	x1,y1 = x0 + width, y0 + height
	cropped_img = im.crop((x0,y0,x1,y1))
	
	hashes.append(hsh)
	cropped_img.save(PROCESSED_IMG_DIR + hsh + ".jpg")

english = list(data['en'])
hindi = list(data['hi'])

modified_data = pd.DataFrame(list(zip(hashes, english, hindi)), columns=['hash','english', 'hindi'])
modified_data.to_csv(CSV_FILENAME+".processed", sep="\t")
