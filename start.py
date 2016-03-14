#!/usr/bin/env python2

import subprocess

import csv
import numpy
from scipy.signal import convolve2d

from sklearn import svm


params = {'gf_psi': 2.859,
          'gf_gamma': 1.703,
          'gf_lambda': 7.086,
          'gf_sigma': 3.765,
          'nu': -4.938,
          'gamma': -1.141}


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

# ------------------------- LOAD DATA


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


# ----------------------- PREPROCESS DATA

print "Normalizing data..."
Xtr /= numpy.linalg.norm(Xtr, 2, axis=1, keepdims=True)
Xte /= numpy.linalg.norm(Xte, 2, axis=1, keepdims=True)

print "Apply gabor filters..."
filters = [gabor_fn(params['gf_sigma'], x, params['gf_lambda'],
                         params['gf_psi'], params['gf_gamma'])
          for x in numpy.arange(0.,2*numpy.pi-0.01, 2*numpy.pi/6)]
def gf(x):
    x = x.reshape((-1, 28, 28))
    x = numpy.concatenate([
            numpy.concatenate([convolve2d(x[i,:,:], f, 'same')[None, :, :]
                               for i in range(x.shape[0])],
                              axis=0)[:, :, :, None]
            for f in filters], axis=3)
    x = x.reshape((-1, 7, 4, 7, 4, len(filters)))
    x = abs(x).mean(axis=4).mean(axis=2)
    x = x.reshape((x.shape[0],-1))
    return x
Xtr = gf(Xtr)
Xte = gf(Xte)

print "Centering features for data..."
xmean = Xte.mean(axis=0, keepdims=True)
Xte -= xmean
Xtr -= xmean

# ------------------------------------- ONE OF VARIOUS CLASSIFIERS

c= svm.NuSVC(kernel='rbf', gamma=numpy.exp(params['gamma']),
             nu=numpy.exp(params['nu']))


# --------------------------- RUN IT


print "Fitting model with with", params
c.fit(Xtr, Ytr[:, 1])

print "Doing classification"
y = c.predict(Xte)

print "Writing Yte"
f = open('Yte.csv', 'w')
w = csv.writer(f)
w.writerow(['Id', 'Prediction'])
for i in range(y.shape[0]):
    w.writerow([i+1, int(y[i])])

# vim: set sts=4 ts=4 sw=4 tw=0 et :
