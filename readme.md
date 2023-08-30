The following codebase is adapted from multiple libraries and data sources that includes:

1. https://github.com/libeineu/fairseq_mmt
2. https://github.com/AI4Bharat/indicTrans
3. https://ufal.mff.cuni.cz/hindi-visual-genome
4. https://ufal.mff.cuni.cz/bengali-visual-genome
5. https://ufal.mff.cuni.cz/malayalam-visual-genome

Please refer to their terms and conditions and copyrights before usage.

## Installation
cd codes
pip install -e .
cd ..

## Download the files
mkdir -p datasets/multimodal/hi
cd datasets/multimodal/hi
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3267{/README.txt,/hindi-visual-genome-train.txt.gz,/hindi-visual-genome-dev.txt.gz,/hindi-visual-genome-test.txt.gz,/hindi-visual-genome-challenge-test-set.txt.gz,/hindi-visual-genome-11.zip}
unzip hindi-visual-genome-11.zip
cd ..

mkdir bn
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3722{/README.txt,/bengali-visual-genome-train.txt.gz,/bengali-visual-genome-dev.txt.gz,/bengali-visual-genome-test.txt.gz,/bengali-visual-genome-challenge-test-set.txt.gz,/bengali-visual-genome-10.zip}
unzip bengali-visual-genome-10.zip
cd .. 

mkdir ml
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3533{/README.txt,/malayalam-visual-genome-train.txt.gz,/malayalam-visual-genome-dev.txt.gz,/malayalam-visual-genome-test.txt.gz,/malayalam-visual-genome-chtest.txt.gz,/malayalam-visual-genome-10.zip}
unzip malayalam-visual-genome-10.zip
cd ..

cd ../codes
wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/en2indic.zip
unzip en2indic.zip

Use get_img_feat.py and get_img_feat_full.py to extract ViT features

bash finetune_preprocess.sh
bash finetune_mmtrans.sh
bash finetune.sh

Use joint_translate.sh to generate and score on the test set. 
Update the file paths accordingly