#!/usr/bin/env python2

import subprocess

import numpy
from sklearn import svm

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
Xtr = numpy.loadtxt('Xtr.csv', delimiter=',')
print "Load Ytr"
Ytr = numpy.loadtxt('Ytr.csv', delimiter=',', skiprows=1)
print "Load Xte"
Xte = numpy.loadtxt('Xte.csv', delimiter=',')

# Normalize
Xmean = Xtr.mean(axis=0, keepdims=True)
Xtr = Xtr - Xmean
Xvar = numpy.sqrt((Xtr**2).mean(axis=0, keepdims=True))
Xtr= Xtr / Xvar

Xte = (Xte - Xmean) / Xvar

# Split training/validation
Xtr, Xva = Xtr[:4500], Xtr[4500:]
Ytr, Yva = Ytr[:4500,1], Ytr[4500:,1]

print "---- GAUSSIAN KERNEL ----"
c = svm.NuSVC(kernel='rbf', gamma=1./784.)
c.fit(Xtr, Ytr)

print "Error rate: ", numpy.not_equal(c.predict(Xva), Yva).mean()

# vim: set sts=0 ts=4 sw=4 tw=0 noet :
