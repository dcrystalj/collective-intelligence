import unittest

#Pred uporabo uvozite svoje resitve. Prilagodite spodnjo vrstico:
from ogrodje import *

class TestLogisticRegression(unittest.TestCase):

    def data1(self):
        X = numpy.array([[ 5.0, 3.6, 1.4, 0.2 ],
                         [ 5.4, 3.9, 1.7, 0.4 ],
                         [ 4.6, 3.4, 1.4, 0.3 ],
                         [ 5.0, 3.4, 1.5, 0.2 ],
                         [ 5.6, 2.9, 3.6, 1.3 ],
                         [ 6.7, 3.1, 4.4, 1.4 ],
                         [ 5.6, 3.0, 4.5, 1.5 ],
                         [ 5.8, 2.7, 4.1, 1.0 ]])
        y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X,y

    def test_h(self):
        X,y = self.data1()
        self.assertAlmostEqual(h(X[0], numpy.array([0,0,0,0])), 0.5)
        self.assertAlmostEqual(h(X[0], numpy.array([0.1,0.1,0.1,0.1])), 0.73497259)
        self.assertAlmostEqual(h(X[0], numpy.array([0.1,0.2,0.1,0.7])), 0.817574476)


    def test_cost_noreg(self):
        X,y = self.data1()
        self.assertAlmostEqual(cost(numpy.array([0,0,0,0]), X, y, 0.0), 0.69314718)
        self.assertAlmostEqual(cost(numpy.array([0.1,-0.1,0.1,0.1]), X, y, 0.0), 0.61189876)
        self.assertAlmostEqual(cost(numpy.array([1,1,1,1]), X, y, 0.0), 5.17501926)
        self.assertAlmostEqual(cost(numpy.array([-1.06,-5.39,7.64,3.79]), X, y, 0.0), 6.152e-06)

    def assertAlmostEqualLists(self, l1, l2, **kwargs):
        self.assertEqual(len(l1), len(l2))
        for a,b in zip(l1,l2):
            self.assertAlmostEqual(a, b, **kwargs)

    def test_grad_noreg(self):
        X,y = self.data1()
        self.assertAlmostEqualLists(grad(numpy.array([0,0,0,0]), X, y, 0.0),
            [-0.231, 0.162, -0.662, -0.256], places=3)
        self.assertAlmostEqualLists(grad(numpy.array([0.1,-0.1,0.1,0.1]), X, y, 0.0),
            [0.561, 0.597, -0.187, -0.115], places=3)
        self.assertAlmostEqualLists(grad(numpy.array([-1.06,-5.39,7.64,3.79]), X, y, 0.0),
            [0., 0., 0., 0.], places=3)

if __name__ == '__main__':
    unittest.main()
