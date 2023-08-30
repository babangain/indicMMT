echo -e "This is in consonance with the others\tअनुरूप" | fairseq-interactive en-indic/final_bin --user-dir model_configs \
  --path en-indic/model/checkpoint_best.pt \
  --bpe fastbpe \
  --bpe-codes en-indic/vocab/bpe_codes.32k.SRC \
  --constraints \
  -s SRC -t TGT \
  --beam 10