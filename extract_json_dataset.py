import json

for i in range(10594):
	print(i+1)
	fp_json = open('cv_set/%s.json' % str(i+1), 'r', encoding='utf8')
	file = json.load(fp_json)
	fp_cv = open('cv_extracted/cvs/%s_cv.txt' % str(i+1), 'w', encoding='utf8', newline='\n')
	fp_skills = open('cv_extracted/skills/%s_skills.txt' % str(i+1), 'w', encoding='utf8', newline='\n')
	if file["content"] is not None:
		fp_cv.write(file["content"])
		fp_cv.flush()
		fp_cv.close()

	if file["annotation"] is not None:
		for annotation in file["annotation"]:
			if 'Skills' in annotation["label"]:
				for point in annotation["points"]:
					fp_skills.write(point["text"])
					fp_skills.write('\n')
	fp_skills.flush()
	fp_skills.close()
	fp_json.close()
