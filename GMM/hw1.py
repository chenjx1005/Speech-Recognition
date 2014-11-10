#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

from gmm import GMM

TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"
LOG_FILE = "log.txt"
RESULT_FILE = "result.txt"

def main():
	#log = open(LOG_FILE, "w")
	#sys.stdout = log
	#use train data to train GMM
	points = np.genfromtxt(TRAIN_FILE, usecols=(0,1))
	label = np.genfromtxt(TRAIN_FILE, usecols=2)
	w1 = (label == 1)

	#initialize 2 GMMs
	gmm_1 = GMM(4,2)
	gmm_2 = GMM(4,2)

	#train
	print "---------------GMM_1------------------"
	gmm_1.train(points[w1])
	print gmm_1
	print "---------------GMM_2------------------"
	gmm_2.train(points[~w1])
	print gmm_2

	#visualization 2 GMMs
	#plt.figure()
	#gmm_1.draw()
	#plt.show()

	#plt.figure()
	#gmm_2.draw()
	#plt.show()

	#use dev data to classify and compute accuracy
	devs = np.genfromtxt(DEV_FILE, usecols=(0,1))
	dev_la = np.genfromtxt(DEV_FILE, usecols=2)
	re = (dev_la == 1)

	p_1 = gmm_1.predict(devs)
	p_2 = gmm_2.predict(devs)
	pr = ((p_1 > p_2) == re)
	accuracy = np.count_nonzero(pr) * 1.0 / len(pr) 
	print "dev data classify accuracy is", accuracy

	#use test data to classify
	tests = np.genfromtxt(TEST_FILE, usecols=(0,1))
	t_1 = gmm_1.predict(tests)
	t_2 = gmm_2.predict(tests)

	result = [1 if t_1[i] > t_2[i] else 2 for i in range(len(t_1))]
	f = open(RESULT_FILE, "w")
	for i in range(len(tests)):
		line = "%.6f %.6f  %d\n" % (tests[i, 0], tests[i, 1], result[i])
		f.write(line)
	f.close()

if __name__ == '__main__':
	main()