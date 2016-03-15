#!/usr/bin/env python2

import subprocess
import sys
import os
import random
import time
import csv
import numpy
from scipy.signal import convolve2d
from sklearn import svm, decomposition, discriminant_analysis, cross_validation
import matplotlib.pyplot as plt
import mySVM as mysvm

from Oscar import Oscar


# Gabor filters
def gabor_fn(sigma,theta,Lambda,psi,gamma, r=5):
    sigma_x = sigma
    sigma_y = float(sigma)/gamma
    
    # Bounding box
    xmin, xmax, ymin, ymax = -r,r,-r,r
    (x,y) = numpy.meshgrid(numpy.arange(xmin,xmax+1),numpy.arange(ymin,ymax+1 ))
    
    # Rotation 
    x_theta=x*numpy.cos(theta)+y*numpy.sin(theta);
    y_theta=-x*numpy.sin(theta)+y*numpy.cos(theta);
    
    gb= numpy.exp(-.5*(x_theta**2/sigma_x**2+y_theta**2/sigma_y**2))*numpy.cos(2*numpy.pi/Lambda*x_theta+psi);
    return gb



scientist = Oscar('Brsw0L6x4vxebEDU9BNVySwGDDAYEqgd0kXE0OGKag9zfnNlbnNvdXQtb3NjYXJyEQsSBFVzZXIYgICAgLrGgQoM')

# Check we have the files
files = [
    ('Xtr.csv', '87a2ddf60ee89a00f9be81ab1d896820'),
    ('Ytr.csv', 'c6a430b0f8be7353ef275923e96b6d18'),
    ('Xte.csv', 'd775122d4a535e569bc20a653e1cfc02')
]

for f, md5 in files:
    val = subprocess.Popen(['md5sum', f], stdout=subprocess.PIPE).communicate()[0].split(' ')[0]
    if val != md5:
        print "Please download %s to this folder" % f

# Load data
print "Load Xtr"
Xtrva = numpy.loadtxt('Xtr.csv', delimiter=',')
print "Load Ytr"
Ytrva = numpy.loadtxt('Ytr.csv', delimiter=',', skiprows=1)
print "Load Xte"
Xte = numpy.loadtxt('Xte.csv', delimiter=',')


exp = {'name': ' '.join(sys.argv)}


# ------------------------------------- ONE OF VARIOUS CLASSIFIERS

if "--kpca" in sys.argv:
    exp['parameters'] = {
           'n_pc': {"min": 2, "max": 100,"step":1},
           'gamma': {"min": numpy.log(1./784. * 1e-2),
                     "max": numpy.log(1./784. * 1e4)},
       }
    class Classifier:
        def __init__(self, job):
            self.pca = decomposition.KernelPCA(n_components=job['n_pc'], 
                                               kernel='rbf',
                                               gamma=numpy.exp(job['gamma']))
            self.cl = discriminant_analysis.QuadraticDiscriminantAnalysis()
        def fit(self, x, y):
            xtr = self.pca.fit_transform(x)
            self.cl.fit(xtr, y)
        def predict(self, x):
            xtr = self.pca.transform(x)
            return self.cl.predict(xtr)
    classifier = Classifier

if "--gaussian" in sys.argv:
    exp['parameters'] = {
           'gamma': {"min": numpy.log(1./784. * 1),
                     "max": numpy.log(1./784. * 1e6)},
           'nu': {"min": numpy.log(1./4500. * 1),
                  "max": numpy.log(1./4500. * 100)},
       }
    classifier = (lambda job:
        svm.NuSVC(kernel='rbf', gamma=numpy.exp(job['gamma']),
                                nu=numpy.exp(job['nu']),
                                decision_function_shape='ovo')
    )

if "--gaussian-nusvm" in sys.argv:
    exp['parameters'] = {
           'gamma': {"min": numpy.log(1./784. * 1),
                     "max": numpy.log(1./784. * 1e6)},
           'nu': {"min": numpy.log(1./4500. * 1),
                  "max": numpy.log(1./4500. * 100)},
       }
    classifier = (lambda job:
        mysvm.NuSVM(kernel='rbf', gamma=numpy.exp(job['gamma']),
                                nu=numpy.exp(job['nu']))
    )

if "--gaussian-csvm" in sys.argv:
    exp['parameters'] = {
           'gamma': {"min": numpy.log(1./784. * 1),
                     "max": numpy.log(1./784. * 1e6)},
           'c': {"min": numpy.log(1),
                 "max": numpy.log(1000)},
       }
    classifier = (lambda job:
        mysvm.CSVM(kernel='rbf', gamma=numpy.exp(job['gamma']),
                                C=numpy.exp(job['c']))
    )

if "--poly" in sys.argv:
    exp['parameters'] = {
               'r': {"min": 0.5, "max": 2.},
               'nu': {"min": numpy.log(1./4500. * 0.01),
                      "max": numpy.log(1./4500. * 1000)},
           }
    classifier = (lambda job:
            svm.NuSVC(kernel='poly', degree=2, coef0=job['r'],
                      nu=numpy.exp(job['nu']),
                      decision_function_shape='ovr', cache_size=2000)
    )

if "--linear" in sys.argv:
    exp['parameters'] = {
               'C': {"min": numpy.log(0.1),
                     "max": numpy.log(10.)}
           }

    classifier = (lambda job:
                svm.LinearSVC(C=numpy.exp(job['C']), dual=False, tol=1e-3)
    )


# ----------------------- Add various normalization techniques

for pp in reversed(sys.argv):
    if pp == "--normalize-ex":
        def add_normalize(cl):
            class No:
                def __init__(self, job):
                    self.cl = cl(job)
                def fit(self, x, y):
                    print "Normalizing training data..."
                    xnorm = numpy.linalg.norm(x, 2, axis=1, keepdims=True)
                    self.cl.fit(x / xnorm, y)
                def predict(self, x):
                    xnorm = numpy.linalg.norm(x, 2, axis=1, keepdims=True)
                    return self.cl.predict(x / xnorm)
            return No
        classifier = add_normalize(classifier)

    if pp == "--center-feature":
        def add_cf(cl):
            class CF:
                def __init__(self, job):
                    self.cl = cl(job)
                def fit(self, x, y):
                    print "Centering features for training data..."
                    self.xmean = x.mean(axis=0, keepdims=True)
                    self.cl.fit(x - self.xmean, y)
                def predict(self, x):
                    return self.cl.predict(x - self.xmean)
            return CF
        classifier = add_cf(classifier)

    if pp == "--gabor":
        exp['parameters'].update({
            'gf_sigma': {"min": 0.2, "max": 5},
            'gf_lambda': {"min": 1, "max": 10},
            'gf_psi': {"min": -numpy.pi, "max": numpy.pi},
            'gf_gamma': {"min": 0.5, "max": 2},
        })
        def add_gabor(cl):
            class GaborPreClassifier:
                def __init__(self, job):
                    self.cl = cl(job)
                    self.filters = [gabor_fn(job['gf_sigma'], x, job['gf_lambda'],
                                             job['gf_psi'], job['gf_gamma'])
                                    for x in numpy.arange(0.,2*numpy.pi-0.01, 2*numpy.pi/6)]
                def transform(self, x):
                    x = x.reshape((-1, 28, 28))
                    x = numpy.concatenate([
                            numpy.concatenate([convolve2d(x[i,:,:], f, 'same')[None, :, :]
                                               for i in range(x.shape[0])],
                                              axis=0)[:, :, :, None]
                            for f in self.filters], axis=3)
                    x = x.reshape((-1, 7, 4, 7, 4, len(self.filters)))
                    x = abs(x).mean(axis=4).mean(axis=2)
                    x = x.reshape((x.shape[0],-1))
                    return x
                def fit(self, x, y):
                    print "Passing training data through Gabor filter bank..."
                    self.cl.fit(self.transform(x), y)
                def predict(self, x):
                    return self.cl.predict(self.transform(x))
            return GaborPreClassifier
        classifier = add_gabor(classifier)


# --------------------------- RUN IT

best = None
best_p = None

try:
    for i in range(100):
        try:
            job = scientist.suggest(exp)
            time.sleep(2)
        except:
            break
        print job

        idx = list(range(Xtrva.shape[0]))
        random.shuffle(idx)

        split = range(0, len(idx), len(idx)/5) + [len(idx)]
        score = []
        for i in range(1, len(split)):
            a, b = split[i-1:i+1]
            train, valid = idx[:a]+idx[b:], idx[a:b]
            c = classifier(job)
            c.fit(Xtrva[train,:], Ytrva[train,1])
            score.append(numpy.not_equal(c.predict(Xtrva[valid,:]), Ytrva[valid,1]).mean())
            print a, b, score[-1]
            if score[-1] > 0.2: break # gain time
        error_rate = numpy.array(score).mean()

        print "   error_rate=", error_rate
        scientist.update(job, {'loss': error_rate})
        if best == None or error_rate < best:
            best_p = job
            best = error_rate
except KeyboardInterrupt:
    pass

print "Fitting model with with", best_p
c = classifier(best_p)
c.fit(Xtrva, Ytrva[:, 1])
y = c.predict(Xte)

print "Doing final classification"
f = open('Yte-%02.4f.csv'%(best*100), 'w')
w = csv.writer(f)
w.writerow(['Id', 'Prediction'])
for i in range(y.shape[0]):
    w.writerow([i+1, int(y[i])])

# vim: set sts=4 ts=4 sw=4 tw=0 et :
