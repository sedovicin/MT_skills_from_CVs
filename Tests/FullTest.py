import unittest
import TextCleaningTool as tct
import TextSegmentator as ts
import SentenceTokenizator as st
import POSTagger as post
import TextNormalizator as tn

class FullTest(unittest.TestCase):
	def test_full(self):
		text = tct.run('testFiles/small.pdf')
		sentences = ts.run(text)
		tokens = st.run(sentences)
		self.assertEqual(RESULT, tokens)
		words_pos_tags = post.run(tokens)

		print(words_pos_tags)
		for sentence_tokens in words_pos_tags:
			for token in sentence_tokens:
				print(token[0] + ":")
				print("Stem:" + tn.stem(token[0]))
				print("Lemmatized:" + tn.lemmatize(token))


if __name__ == '__main__':
	unittest.main()

RESULT = [["This", "is", "a", "small", "test", "PDF", "."], ["Hello", "!"]]
