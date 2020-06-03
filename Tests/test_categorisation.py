import unittest
from categorisators import CategorisatorNN


class MyTestCase(unittest.TestCase):
	def test_something(self):
		x_train = ["Hello", "it's", "me", "Marcus", "What", "up", "This", "is", "Sparta"]
		y_train = [0,0,0,1,0,0,0,0,1]
		x_test = ["Hello", "Marko", "up", "down"]
		y_test = [0,1,0,0]
		categorisator = CategorisatorNN()
		categorisator.train(x_train, y_train)
		categorisator.evaluate(x_test, y_test)


if __name__ == '__main__':
	unittest.main()
