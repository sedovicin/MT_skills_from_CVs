import pdfminer.high_level as pdf2txt;
from pathlib import Path;
import os;


def run(file: str):
	# TODO: check file type instead of file extension
	extension = Path(file).suffix.lower();
	if (os.path.isfile(file) == False):
		raise FileNotFoundError();
	if (extension == '.pdf'):
		return extractFromPDF(file);
	elif (extension == '.docx'):
		return extractFromDocx(file);
	elif (extension == '.doc'):
		return extractFromDoc(file);
	elif (extension == '.txt'):
		return extractFromTxt(file);
	else:
		raise TypeError("Unsupported file type!");


def extractFromPDF(pdfFile):
	return pdf2txt.extract_text(pdfFile);

def extractFromDocx(docxFile):
	raise Exception("Not implemented extractFromDocx");

def extractFromDoc(docFile):
	raise Exception("Not implemented extractFromDoc");

def extractFromTxt(txtFile):
	raise Exception("Not implemented extractFromTxt");