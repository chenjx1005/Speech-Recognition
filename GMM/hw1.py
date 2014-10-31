#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

from gmm import GMM

TRAIN_FILE = "train.txt"

def main():
	points = np.genfromtxt(TRAIN_FILE, usecols=(0,1))
	label = np.genfromtxt(TRAIN_FILE, usecols=2)
	w1 = (label == 1)
	# plt.plot(points[w1,0], points[w1,1], 'b.', points[~w1,0], points[~w1,1], 'r.')
	# kmeans
	centroids, la = kmeans2(points[w1], 4, 20)
	plt.plot(points[la==0,0], points[la==0,1], 'b.')
	plt.plot(points[la==1,0], points[la==1,1], 'g.')
	plt.plot(points[la==2,0], points[la==2,1], 'y.')
	plt.plot(points[la==3,0], points[la==3,1], 'c.')
	# plt.plot(centroids[:,0], centroids[:,1], 'ro')
	mygmm = GMM(4,2)
	mygmm.train(points[w1])
	plt.plot(mygmm.u[:,0], mygmm.u[:,1], 'ro')
	p = mygmm.predict(points[w1])
	print p
	plt.show()

if __name__ == '__main__':
	main()