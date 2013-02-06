import pylab
import math
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
    return 1.0/(1+numpy.exp(-theta.dot(x.T)))

def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    m=y.size
    J=-sum(1.0*y * numpy.log(h(X,theta)) + (1 - y) * numpy.log(1 - h(X,theta))/m + (lambda_/(2*m))*sum(theta**2))
    #return J
    return    numpy.add(numpy.divide(numpy.multiply(-1.0, numpy.sum(numpy.add(numpy.multiply(y, numpy.log(h(X, theta))),
        numpy.multiply(numpy.subtract(1.0, y), numpy.log(numpy.subtract(1.0, h(X, theta))))))),
        (len(X))), numpy.multiply((lambda_/(2.0*len(X))), numpy.sum(numpy.power(theta,2.0))))



def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    m=y.size
    #ones=numpy.identity(theta.shape)
    #ones[0][0]=0
    return(-((y-h(X,theta)).dot(X))/m) #-(lambda_/m)*theta

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


def draw_decision(X, y, classifier, at1, at2, grid=50):

    points = numpy.take(X, [at1, at2], axis=1)
    maxx, maxy = numpy.max(points, axis=0)
    minx, miny = numpy.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    for c,(x,y) in zip(y,points):
        pylab.text(x,y,str(c), ha="center", va="center")
        pylab.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = numpy.zeros([num, num])
    for xi,x in enumerate(numpy.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(numpy.linspace(miny, maxy, num=num)):
            #probability of the closest example
            diff = points - numpy.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = numpy.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pylab.imshow(prob, extent=(minx,maxx,maxy,miny))

    pylab.xlim(minx, maxx)
    pylab.ylim(miny, maxy)
    pylab.xlabel(at1)
    pylab.ylabel(at2)

    pylab.show()

def test_cv(learner, X, y, k=5):
    j=0; e=0;
    vector= []#numpy.zeros(len(y))
    ratio = 1.*len(X)/k

    #k-cross
    for e in range(k):

        newX,testX,newY,testY = [],[],[],[]

        #divide for learning and testing
        for i in range(len(X)):
            if(e*ratio > i or (e+1)*ratio<=i):
                newX.append(X[i])
                newY.append(y[i])
            else:
                testX.append(X[i])
                testY.append(y[i])

        classifier = learner(numpy.asarray(newX), numpy.asarray(newY))

        for i,el in enumerate(testX):
            i+=e*ratio
            cli = classifier(el)
            vector.append(cli)
    """            print(cl,"  ",y[i])
            prediction = 0;
            if(cl[1]>cl[0]):
                if(y[i]==1):
                    vector[i]=cl[1]
                else:
                    vector[i]=1-cl[1]
            else:
                if(y[i]==0):
                    vector[i]=cl[0]
                else:
                    vector[i]=1-cl[0]
     #       print(vector[i])
    #vector = numpy.concatenate((vector,napoved))
    #print(sum(vector)/len(vector))"""
    return numpy.asarray(vector)

def test_learning(learner, X, y):
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results

def CA(y, res):
    truk = [ 1 if y[i]== res[i] else 0 for i in range(len(y)) ]
    return sum(truk)/len(y)

X,y = load('reg.data.txt')

learner = LogRegLearner(lambda_=0.00000)
classifier = learner(X,y) #dobimo model

napoved = classifier(X[0]) #napoved za prvi primer
print(napoved)

#draw_decision(X,y,classifier,0,1)

res = test_cv(learner, X, y)
print("aaa",res,y,len(y), len(X))

#print("Tocnost:", CA(y, res))
