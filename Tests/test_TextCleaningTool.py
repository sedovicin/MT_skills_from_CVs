from unittest import TestCase
import TextCleaningTool as tct;

class Test(TestCase):
	def test_run(self):
		with self.assertRaises(FileNotFoundError):
			tct.run('testFiles/nepostojeci.bzvz');
		with self.assertRaises(TypeError):
			tct.run('testFiles/cv.html');
		self.assertEqual(tct.run('testFiles/small.pdf'), """This is a small test PDF. 

 

Hello! 

""");
		self.assertEqual(tct.run('testFiles/small.docx'), """This is a small test DOCX. 

Hello! 
""");
		self.assertEqual(tct.run('testFiles/small.doc'), """This is a small test DOC. 

Hello! 
""");



