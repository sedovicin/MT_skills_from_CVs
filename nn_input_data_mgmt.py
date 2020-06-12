import dataset_corpus_mgmt as dc_mgmt
import POSTagger
import nltk
from nltk.tree import Tree


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


class Phrase:
	"""
	Structure for holding phrase along with its context and skill category.
	"""
	def __init__(self, pre_phrase_context, phrase, post_phrase_context, category):
		self.pre_phrase_context = pre_phrase_context
		self.phrase = phrase
		self.post_phrase_context = post_phrase_context
		self.category = category

	def str(self):
		return "(%s; %s; %s; %s)" % (self.pre_phrase_context, self.phrase, self.post_phrase_context, self.category)

	def __str__(self):
		return self.str()

	def __repr__(self):
		return self.str()


def get_phrases(cv_path, words):
	"""
	Extracts phrases from file containing CV.
	Requires dictionary to determine into which category the word (and phrase) belong.
	:param cv_path: path to file to be processed
	:type cv_path: str
	:param words: dictionary containing words as keys and values as categories
	:type words: dict[str, int]
	:return: list containing list of phrases for every sentence
	:rtype: list[list[Phrase]]
	"""
	cv_sentences = dc_mgmt.file_to_tokens(cv_path)
	tags = POSTagger.tag_pos_sentences(cv_sentences)
	parser = PhraseParser()
	parsed_sents = parser.parse_sents(tags)

	#TODO: try with ignoring punctuation
	cv_phrases = list()
	for sentence in parsed_sents:
		sent_phrases = list()
		i = 0
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
				pre_phrase_context.append(sent_with_parsed_tags[i][0][0])
				i -= 1
			pre_phrase_context.reverse()

			# GET POST-PHASE CONTEXT
			post_phrase_context = list()
			i = j
			while (i < len(sent_with_parsed_tags)) and ((i - j) < 3):
				post_phrase_context.append(sent_with_parsed_tags[i][0][0])
				i += 1

			# GET PHRASE
			phrase = list()
			i = begin
			while i < j:
				phrase.append(sent_with_parsed_tags[i][0][0])
				i += 1
			new_phrase = Phrase(pre_phrase_context, phrase, post_phrase_context, category)
			sent_phrases.append(new_phrase)

			# Moving on...
			i = j
		cv_phrases.append(sent_phrases)
	return cv_phrases

def create_sets(start, end):
	"""
	Creates sets from files and exports them to files
	"""
	all_phrases = list()
	for i in range(start, end):
		categories = dict()

		sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/%s_skills.txt' % (i+1))
		for sentence in sentences_skills:
			for word in sentence:
				categories[word] = 1
		cv_phrases = get_phrases('cv_extracted/cvs/%s_cv.txt' % (i+1), categories)
		all_phrases.extend(cv_phrases)


def main():
	categories = dict()

	sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/1_skills.txt')
	for sentence in sentences_skills:
		for word in sentence:
			categories[word] = 1
	cv_phrases = get_phrases('cv_extracted/cvs/1_cv.txt', categories)
	print(cv_phrases)


if __name__ == "__main__":
	main()