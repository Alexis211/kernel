#!/usr/bin/env python2

import subprocess
import sys

import numpy
from sklearn import svm

from Oscar import Oscar

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

# Normalize

for pp in sys.argv:
    if pp == "--normalize-ex":
        Xnorm = numpy.linalg.norm(Xtrva, 2, axis=1, keepdims=True)
        Xtrva = Xtrva / Xnorm

        Xte_norm = numpy.linalg.norm(Xte, 2, axis=1,keepdims=True)
        Xte = Xte / Xte_norm

    if pp == "--center-feature":
        Xmean = Xtrva.mean(axis=0, keepdims=True)
        Xtrva = Xtrva - Xmean
        Xte = Xte - Xmean

# Split training/validation
Xtr, Xva = Xtrva[:4500], Xtrva[4500:]
Ytr, Yva = Ytrva[:4500,1], Ytrva[4500:,1]

exp = {'name': ' '.join(sys.argv)}


# ------------------------------------- ONE OF VARIOUS CLASSIFIERS

if "--gaussian" in sys.argv:
    exp['parameters'] = {
           'gamma': {"min": numpy.log(1./784. * 1),
                     "max": numpy.log(1./784. * 1e6)},
           'nu': {"min": numpy.log(1./4500. * 1),
                  "max": numpy.log(1./4500. * 100)},
       }
    classifier = (lambda job:
        svm.NuSVC(kernel='rbf', gamma=numpy.exp(job['gamma']),
                                nu=numpy.exp(job['nu']))
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

# --------------------------- RUN IT

while True:
    try:
        job = scientist.suggest(exp)
    except:
        break
    print job
    c = classifier(job)
    c.fit(Xtr, Ytr)
    error_rate = numpy.not_equal(c.predict(Xva), Yva).mean()
    print "error_rate=", error_rate
    scientist.update(job, {'loss': error_rate})


# vim: set sts=4 ts=4 sw=4 tw=0 et :
