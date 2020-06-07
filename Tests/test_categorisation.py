import unittest
from categorisators import CategorisatorNN
import json

class MyTestCase(unittest.TestCase):
	def test_something(self):
		x_train = []
		y_train = []
		with open('../dataset.json', 'r', encoding='utf8') as fp:
			file = json.load(fp)
			for item in file.items():
				x_train.append(item[0])
				y_train.append(item[1])

		x_test = ["C++", "Marko", "path", "ball", "machine"]
		y_test = [1,0,0,0,1]
		categorisator = CategorisatorNN('corpus.json')

		categorisator.train(x_train, y_train)
		categorisator.evaluate(x_test, y_test)
		categorisator.predict(x_test)


if __name__ == '__main__':
	unittest.main()
