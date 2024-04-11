import unittest

def is_even(num):
    return num % 2 == 0

class TestEvenNumber(unittest.TestCase):
    def test_even_number(self):
        # Test if 4 is an even number
        self.assertTrue(is_even(4))

    def test_odd_number(self):
        # Test if 5 is an even number
        self.assertFalse(is_even(5))

if __name__ == '__main__':
    unittest.main()
