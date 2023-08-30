import pandas as pd
from tqdm import tqdm
import langid
from rich.progress import track
from PIL import Image
langid.set_languages(['en','ml','bn','hi'])
# for lang_folders in ['ml', 'hi','bn']:
split_to_filename = {
	'train':'train', 'valid': 'dev', 'test': 'test', 'challenge': 'challenge-test-set'
}
lang_short_to_full = {'bn':'bengali','hi':'hindi','ml':'malayalam'}

def prep_data(langs=['ml','bn','hi'],subsets=['train','test','valid','challenge'],folder_name=""):
	for subset in subsets:
		for lang in langs:
			ip_filename = lang+"/"+lang_short_to_full[lang]+ "-visual-genome-"+ split_to_filename[subset]+".txt"
			print("Reading data for ", subset,lang,ip_filename)

			data = pd.read_csv(ip_filename,sep="\t",header=None,names=['img','x1','y1','x2','y2','src','tgt'])
			if subset == 'train':
				try:
					os.mkdir('train/en-'+lang)
				except:
					pass
			outfile=None
			if subset=='train':
				outfile='train/en-'+lang+"/"+"train."
			else:
				outfile=subset+"/"+subset+"."

			print("Running Langid")
			for idx in track(range(len(data))):
				english = data['src'].iloc[idx]
				target = data['tgt'].iloc[idx]
				if langid.classify(english)[0] != "en":
					print("Non English Found at, ",idx, "Sent: ",english)
				try:
					if langid.classify(target)[0] != lang:
						print("Non target Found at, ",idx, "Sent: ",target)
				except:
					print(idx,target)

			data['src'].to_csv(outfile+"en",header=None, index=None)
			data['tgt'].to_csv(outfile+lang,header=None, index=None)

			#To verify if they contain same image for all lang
			data['img'].to_csv('files/'+subset+"."+lang,header=None,index=None)


def prep_data_img(langs=['ml'],subsets=['test','train','valid','challenge'],folder_name=""):
	for subset in subsets:
		for lang in langs:
			ip_filename = lang+"/"+lang_short_to_full[lang]+ "-visual-genome-"+ split_to_filename[subset]+".txt"
			print("Reading data for ", subset,lang,ip_filename)
			IMG_DIR="ml/malayalam-visual-genome-10/malayalam-visual-genome-"+split_to_filename[subset]+".images/"
			data = pd.read_csv(ip_filename,sep="\t",header=None,names=['img','x1','y1','x2','y2','src','tgt'])
			f = open(subset+"/"+subset+"_files.txt",'w')
			g = open(subset+"/"+subset+"_files_crop.txt",'w')

			for i in tqdm(range(len(data))):
				# print(data.iloc[i])
				imgID, x0, y0, width, height, en, hi = data.iloc[i,:].tolist()
				# print(imgID,x0,y0)
				im = Image.open(IMG_DIR+str(imgID)+".jpg")
				x1,y1 = x0 + width, y0 + height
				cropped_img = im.crop((x0,y0,x1,y1))
				cropped_img.save(IMG_DIR +str(imgID)  + "_crop.jpg")
				f.write(str(imgID)+".jpg\n")
				g.write(str(imgID)+"_crop.jpg\n")

prep_data()
prep_data_img(subsets=['test','valid','challenge'])