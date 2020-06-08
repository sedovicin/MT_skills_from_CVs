from unittest import TestCase
import TextFromFileExtractor as extractor;


class Test(TestCase):
	def test_run(self):
		with self.assertRaises(FileNotFoundError):
			extractor.extract('testFiles/not_existing.file')
		with self.assertRaises(TypeError):
			extractor.extract('testFiles/cv.html')

		self.assertEqual(PDF_FILE_TEXT, extractor.extract('testFiles/small.pdf'))
		self.assertEqual(TXT_FILE_TEXT, extractor.extract('testFiles/small.txt'))
		self.assertEqual(DOCX_FILE_TEXT, extractor.extract('testFiles/small.docx'))
# 		self.assertEqual(DOC_FILE_TEXT, tct.run('testFiles/small.doc'));


PDF_FILE_TEXT = """This is a small test PDF. 

 

Hello! 

"""
TXT_FILE_TEXT = """This is a small test TXT.

Hello!
"""
DOCX_FILE_TEXT = """This is a small test DOCX.

Hello!
"""
DOC_FILE_TEXT = """This is a small test DOC. 

Hello! 
"""
