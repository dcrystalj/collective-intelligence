import unittest

#Pred uporabo uvozite svoje resitve. Prilagodite spodnjo vrstico:
from ogrodje import *

class DummyCVLearner:
    """ For CV testing """
    def __call__(self,X,y):
        return DummyCVClassifier(ldata=X)

class TestirasNaUcnihPodatkih(Exception): pass
class DeliNisoEnakoVeliki(Exception): pass
class VsiPrimeriNisoTestirani(Exception): pass
class PremesanVrstniRed(Exception): pass

class DummyCVClassifier:
    def __init__(self, ldata):
        self.ldata = map(list, ldata)
    def __call__(self, x):
        if list(x) in self.ldata:
            raise TestirasNaUcnihPodatkih()
        else:
            return [sum(x), len(self.ldata)]

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

    def test_ca(self):
        X,y = self.data1()
        self.assertAlmostEqual(CA(y, [[1,0]]*len(y)), 0.5)
        self.assertAlmostEqual(CA(y, [[0.5,1]]*len(y)), 0.5)
        self.assertAlmostEqual(CA(y, [[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0]]), 0.0)

    def test_logreg_noreg_learning_ca(self):
        X,y = self.data1()
        logreg = LogRegLearner(lambda_ = 0)
        pred = test_learning(logreg, X, y)
        ca = CA(y, pred)
        self.assertAlmostEqual(ca, 1.)

    def test_cv(self):
        X,y = self.data1()

        pred = test_cv(DummyCVLearner(),X,y,k=4)
        #pred = test_learning(DummyCVLearner(),X,y)
        if len(set([a for _,a in pred])) != 1:
            raise DeliNisoEnakoVeliki()

        signatures = [a for a,_ in pred]
        if len(set(signatures)) != len(y):
            raise VsiPrimeriNisoTestirani()
    
        if signatures != map(lambda x: sum(list(x)), X):
            raise PremesanVrstniRed()

if __name__ == '__main__':
    unittest.main()
