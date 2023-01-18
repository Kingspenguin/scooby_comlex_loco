import unittest

from a1_utilities.cpg.gait_generator import GaitGenerator

class TestStringMethods(unittest.TestCase):

    def test_constructor(self):
        g = GaitGenerator("cuda:0")

if __name__ == '__main__':
    unittest.main()