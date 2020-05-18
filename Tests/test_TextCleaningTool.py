from unittest import TestCase
import TextCleaningTool as tct;


class Test(TestCase):
	def test_run(self):
		with self.assertRaises(FileNotFoundError):
			tct.run('testFiles/not_existing.file')
		with self.assertRaises(TypeError):
			tct.run('testFiles/cv.html')
		self.assertEqual(PDF_FILE_TEXT, tct.run('testFiles/small.pdf'))

		self.assertEqual(TXT_FILE_TEXT, tct.run('testFiles/small.txt'))
		self.assertEqual(DOCX_FILE_TEXT, tct.run('testFiles/small.docx'))
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