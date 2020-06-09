import json
import TextCleaningTool as text_cleaner
import TextSegmentator as segmetator_to_sentences
import SentenceTokenizator as sentence_to_tokens
import POSTagger
from TextNormalizator import WordTextNormalizator

NOT_SKILL = 0
SKILL = 1


def remove_punctuation(tagged_sentences):
	"""Removes punctuation from list of words with their tags as they are not really necessary in further steps.

	:type tagged_sentences: list[list[tuple]]
	:return: new list without punctuation"""
	clean = list()
	for tagged_sentence in tagged_sentences:
		clean_tagged_sentence = list()
		for word_tag in tagged_sentence:
			if word_tag[1] != '.':
				clean_tagged_sentence.append(word_tag)
		clean.append(clean_tagged_sentence)
	return clean


def add_to_dataset(file, dictionary, value, overwrite=True):
	"""Lemmatizes every word in given file and puts it in given dictionary with given value.
	File must be a path to file, not file pointer. If overwrite is true, overwrites
	existing value with new one.

	:type file: str
	:type dictionary: dict
	:type value: int
	:type overwrite: bool"""
	print("Adding %s to dictionary..." %file)
	cleaned_text = text_cleaner.run(file)
	line_in_sentences = segmetator_to_sentences.run(cleaned_text)
	tokens = sentence_to_tokens.run(line_in_sentences)
	pos_tags_sentences = POSTagger.run(tokens)
	pos_tags_sents_clean = remove_punctuation(pos_tags_sentences)
	normalizator = WordTextNormalizator()
	lemmatized_words = list()
	for sentence in pos_tags_sents_clean:
		lemmatized_words.extend(normalizator.lemmatize_sentence(sentence))
	# print(words)
	for word in lemmatized_words:
		if overwrite or word not in dictionary:
			dictionary[word] = value


def import_dataset(file):
	"""Imports dataset from file. File must be readable by json.

	:type file: str
	:returns: Object the file is parsed into"""
	fp = open(file, 'r', encoding='utf8')
	json_file = json.load(fp)
	fp.close()
	return json_file


def export_dataset(dataset, file):
	"""Exports given dataset to given file.

	:type dataset: dict
	:type file: str"""
	print("Started exporting dataset.")
	fp = open(file, 'w', encoding='utf8')
	json.dump(dataset, fp)
	fp.close()
	print("Finished exporting dataset.")


# dataset = dict()
# for i in range(10000):
# 	add_to_dataset('cv_extracted/cvs/%s_cv.txt' % str(i+1), dataset, NOT_SKILL, overwrite=False)
# 	add_to_dataset('cv_extracted/skills/%s_skills.txt' % str(i+1), dataset, SKILL)
# export_dataset(dataset, 'dataset.json')
# print(dataset)
#import vectorizators

#dataset = import_dataset('dataset.json')
#corpus = vectorizators.get_corpus_sentences()
#corpus = list()
# for i in range(10000):
# 	print(i)
# 	cleaned_text = text_cleaner.run('cv_extracted/cvs/%s_cv.txt' % str(i+1))
# 	line_in_sentences = segmetator_to_sentences.run(cleaned_text)
# 	tokens = sentence_to_tokens.run(line_in_sentences)
# 	corpus.extend(tokens)
#
# fp = open('corpus.json', 'w', encoding='utf8')
# json.dump(corpus, fp)
# fp.close()
#print(corpus)

fp = open('corpus.json', 'r', encoding='utf8')
corpus = json.load(fp)
print(corpus)
print(type(corpus))