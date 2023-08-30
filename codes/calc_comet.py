from evaluate import load
comet_metric = load('comet')

output_prefix="gen_finetune_then_freeze"
for lang in ['hi','bn','ml']:
	for testset in ['test','challenge']:

		src = open("../datasets/dataset/"+testset+"/"+testset+".en")
		tgt = open("../datasets/dataset/"+testset+"/"+testset+"."+lang)
		hypo = None
		if testset=="test":
			hypo = output_prefix+".final."+lang
		elif testset == "challenge":
			hypo = output_prefix + ".final.chal."+lang
		hypo_ = open(hypo)
		source = src.readlines()
		hypothesis = hypo_.readlines()
		reference = tgt.readlines()

		# source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
		# hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
		# reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
		comet_score = comet_metric.compute(predictions=hypothesis, references=reference, sources=source,progress_bar=True)
		print(lang, testset)
		print(comet_score['mean_score'])
