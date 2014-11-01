#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

from gmm import GMM

TRAIN_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"

def main():
	points = np.genfromtxt(TRAIN_FILE, usecols=(0,1))
	label = np.genfromtxt(TRAIN_FILE, usecols=2)
	w1 = (label == 1)

	# plt.plot(points[w1,0], points[w1,1], 'b.', points[~w1,0], points[~w1,1], 'r.')
	
	gmm_1 = GMM(4,2)
	gmm_2 = GMM(4,2)

	gmm_1.train(points[w1])
	gmm_2.train(points[~w1])

	plt.figure()
	plt.plot(points[w1,0], points[w1,1], 'b.')
	plt.plot(gmm_1.u[:,0], gmm_1.u[:,1], 'ro')
	gmm_1.draw()
	plt.show()

	plt.figure()
	plt.plot(points[~w1,0], points[~w1,1], 'b.')
	plt.plot(gmm_2.u[:,0], gmm_2.u[:,1], 'go')
	gmm_2.draw()
	plt.show()

	devs = np.genfromtxt(DEV_FILE, usecols=(0,1))
	dev_la = np.genfromtxt(DEV_FILE, usecols=2)
	re = (dev_la == 1)

	p_1 = gmm_1.predict(devs)
	p_2 = gmm_2.predict(devs)
	pr = ((p_1 > p_2) == re)
	accuracy = np.count_nonzero(pr) * 1.0 / len(pr) 
	print accuracy

if __name__ == '__main__':
	main()