import numpy
from scipy.optimize import fmin_l_bfgs_b

def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = numpy.loadtxt(name)
    X, y = data[:,:-1], data[:,-1].astype(numpy.int)
    return X,y

def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # ... dopolnite
    return 0.

def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    # ... dopolnite
    return 0.

def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    # ... dopolnite
    return None

class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = numpy.hstack(([1.], x))
        p1 = h(x, self.th) #verjetno razreda 1
        return [ 1-p1, p1 ] 

class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = numpy.hstack((numpy.ones((len(X),1)), X))

        #optimizacija
        theta = fmin_l_bfgs_b(cost,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)

#
# Primer uporabe
#

X,y = load('reg.data')

learner = LogRegLearner(lambda_=0.0)
classifier = learner(X,y) #dobimo model

napoved = classifier(X[0]) #napoved za prvi primer
print napoved

