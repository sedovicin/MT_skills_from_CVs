from unittest import TestCase
import TextCleaningTool as tct;

class Test(TestCase):
	def test_run(self):
		with self.assertRaises(FileNotFoundError):
			tct.run('testFiles/nepostojeci.bzvz');
		with self.assertRaises(TypeError):
			tct.run('testFiles/cv.html');
		self.assertEqual(tct.run('testFiles/cv.txt'), "This is a test cv. Hello!");




