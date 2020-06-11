import dataset_corpus_mgmt as dc_mgmt
import POSTagger
import TextFromFileExtractor
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
	def __init__(self, pre_phrase_context, phrase, post_phrase_context, skill):
		self.pre_phrase_context = pre_phrase_context
		self.phrase = phrase
		self.post_phrase_context = post_phrase_context
		self.skill = skill

	def str(self):
		return "(%s; %s; %s; %s)" % (self.pre_phrase_context, self.phrase, self.post_phrase_context, self.skill)

	def __str__(self):
		return self.str()

	def __repr__(self):
		return self.str()


sentences_cv = dc_mgmt.file_to_tokens('cv_extracted/cvs/1_cv.txt')
tags = POSTagger.tag_pos_sentences(sentences_cv)
parser = PhraseParser()
parsed_sents = parser.parse_sents(tags)

sentences_skills = dc_mgmt.file_to_tokens('cv_extracted/skills/1_skills.txt')
skills_dict = dict()
for sentence in sentences_skills:
	for word in sentence:
		skills_dict[word] = 1
skills = list(skills_dict.keys())

for sentence in parsed_sents:
	phrases = list()
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
		skill = (sent_with_parsed_tags[begin][0][0] in skills)

		j = begin

		# FIND END OF PHRASE
		while True:
			j += 1
			if (j >= len(sent_with_parsed_tags)) \
				or (sent_with_parsed_tags[j][1] != 'PHRASE') \
				or (skill != (sent_with_parsed_tags[j][0][0] in skills)):
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
		new_phrase = Phrase(pre_phrase_context, phrase, post_phrase_context, skill)
		phrases.append(new_phrase)

		# Moving on...
		i = j



def check_next(sent_with_parsed_tags, skills, curr_index, curr_skill):
	if (sent_with_parsed_tags[curr_index+1][1] == 'PHRASE'):
		if sent_with_parsed_tags[curr_index+1][0][0] in skills:
			skill = True
		else:
			skill = False
		if curr_skill == skill:
			check_next(sent_with_parsed_tags, skills, curr_index+1, curr_skill)


	# break

# for i in range(1000):
# 	sents = dc_mgmt.file_to_tokens('cv_extracted/skills/%s_skills.txt' %str(i+1))
# 	tags = POSTagger.tag_pos_sentences(sents)
# 	parser = PhraseParser()
# 	parsed = parser.parse_sents(tags)
#
# 	for sent in parsed:
#
# 		# a = sent.subtrees(lambda t: t.label() == 'S')
# 		# for subtree in a:
# 		# 	print(subtree.leaves())
# 		for subt in sent.pos():
# 			print(subt)
# 		print("")


