import nltk
import numpy as np
from nltk.tree import Tree

import POSTagger
import dataset_corpus_mgmt as dc_mgmt
import vectorizators


class PhraseParser:
	"""
	Used for extracting only requested parts of sentence, considering Part-Of-Speech tags.
	"""
	def __init__(self, grammar=None):
		self.grammar = grammar
		if self.grammar is None:
			self.grammar = """
				PHRASE: {<VERB>?<ADJ>*<NOUN>+<VERB>?}
					{<VERB>?<ADJ>+<CONJ>+<ADJ>*<NOUN>+<VERB>?}
				"""
		self.parser = nltk.RegexpParser(self.grammar)

	def parse(self, tagged_sentence):
		"""
		Parses sentence.

		:param tagged_sentence: sentence to be parsed
		:return: Parsed sentence as tree
		:rtype: Tree
		"""
		return self.parser.parse(tagged_sentence)

	def parse_sents(self, tagged_sentences):
		"""
		Parses multiple sentences

		:param tagged_sentences: sentences to be parsed
		:return: Parsed sentence as list of trees
		:rtype: list[Tree]
		"""
		sentences_parsed = list()
		for sentence in tagged_sentences:
			sentences_parsed.append(self.parse(sentence))
		return sentences_parsed


class SampleGenerator:
	def __init__(self, corpus_path, dataset_path):
		self.vectorizator = vectorizators.get_word2vec(word2vec_path='word2vec.obj', corpus_path=corpus_path)
		self.parser = PhraseParser()

	def get_batch_x_y(self, begin, end):
		"""
		Creates samples and its targets from files indexed begin index (incl.) to end index (excl.)
		:param begin:
		:param end:
		:return:
		"""
		pre_contexts = []
		phrases = []
		post_contexts = []
		ys = []
		for i in range(begin, end):
			cv_sentences = dc_mgmt.file_to_tokens('cv_extracted/cvs/%s_cv.txt' % i)
			tags = POSTagger.tag_pos_sentences(cv_sentences)
			parsed_sents = self.parser.parse_sents(tags)

			categories = dict()
			sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/%s_skills.txt' % i)
			for sentence in sentences_skills:
				for word in sentence:
					categories[word] = 1

			for sentence in parsed_sents:
				pre_context, phrase, post_context, y = self.phrases_context_as_vectors(sentence, categories)
				pre_contexts.extend(pre_context)
				phrases.extend(phrase)
				post_contexts.extend(post_context)
				ys.extend(y)
		return np.array(pre_contexts), np.array(phrases), np.array(post_contexts), np.array(ys)

	def phrases_context_as_vectors(self, sentence, words):
		i = 0
		pre_contexts = []
		phrases = []
		post_contexts = []
		ys = []
		sent_with_parsed_tags = sentence.pos()
		while i < len(sent_with_parsed_tags):
			# FIND NEXT PHRASE INDEX
			begin = i
			while (begin < len(sent_with_parsed_tags)) and (sent_with_parsed_tags[begin][1] != 'PHRASE'):
				begin += 1
			if begin >= len(sent_with_parsed_tags):
				break

			# GET IF FIRST WORD OF PHRASE IS SKILL OR NOT
			category = words.get(sent_with_parsed_tags[begin][0][0], 0)

			j = begin

			# FIND END OF PHRASE
			while True:
				j += 1
				if (j >= len(sent_with_parsed_tags)) \
					or (sent_with_parsed_tags[j][1] != 'PHRASE') \
					or (category != words.get(sent_with_parsed_tags[begin][0][0], 0)):
					break

			# GET PRE-PHASE CONTEXT
			pre_phrase_context = list()
			i = begin - 1
			while (i >= 0) and ((begin - i) <= 3):
				try:
					pre_phrase_context.append(self.vectorizator.wv.get_vector(sent_with_parsed_tags[i][0][0]))
				except KeyError:
					pre_phrase_context.append(np.zeros(self.vectorizator.wv.vector_size))
				# TODO: generate vector!
				i -= 1
			pre_phrase_context.reverse()

			# GET POST-PHASE CONTEXT
			post_phrase_context = list()
			i = j
			while (i < len(sent_with_parsed_tags)) and ((i - j) < 3):
				try:
					post_phrase_context.append(self.vectorizator.wv.get_vector(sent_with_parsed_tags[i][0][0]))
				except KeyError:
					post_phrase_context.append(np.zeros(self.vectorizator.wv.vector_size))
				# TODO: generate vector!
				i += 1

			# GET PHRASE
			phrase = list()
			i = begin
			while i < j:
				try:
					phrase.append(self.vectorizator.wv.get_vector(sent_with_parsed_tags[i][0][0]))
				except KeyError:
					phrase.append(np.zeros(self.vectorizator.wv.vector_size))
				# TODO: generate vector!
				i += 1

			pre_contexts.append(np.array(pre_phrase_context))
			phrases.append(np.array(phrase))
			post_contexts.append(np.array(post_phrase_context))

			y_entry = np.zeros(2)
			y_entry[category] = 1
			ys.append(y_entry)

			# Moving on...
			i = j
		return pre_contexts, phrases, post_contexts, ys


def main():
	pass


if __name__ == "__main__":
	main()
