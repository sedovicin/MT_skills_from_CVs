import unittest
import TextCleaningTool as tct
import TextSegmentator as ts
import SentenceTokenizator as st


class FullTest(unittest.TestCase):
	def test_full(self):
		text = tct.run('testFiles/small.pdf')
		sentences = ts.run(text)
		tokens = st.run(sentences)
		self.assertEqual(RESULT, tokens)


if __name__ == '__main__':
	unittest.main()

RESULT = [["This", "is", "a", "small", "test", "PDF", "."], ["Hello", "!"]]
