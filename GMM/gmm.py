#!usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr 
from scipy.cluster.vq import kmeans2, ClusterError
from scipy.stats import multivariate_normal

class GMM(object):
	"""
	The Gaussian Mixture Model.

	Parameters
	----------
	k : int
		the number of the models

	dim : int
		the dimension of the data

	pi : ndarray, optional
		mixing coefficients, a k dimension array

	u : ndarray, optional
		the mean value of each Gaussian model, k by d array

	sigma : ndarray, optional
		the covariance matrix of each Gaussian model, k by d by d array
	"""
	def __init__(self, k, dim, pi=None, u=None, sigma=None):
		self.k = int(k)
		self.dim = int(dim)
		if self.k < 1 or self.dim < 1:
			raise ValueError("k and dim must be at least 1")
		#TODO: varify the parameters pi, u, sigma
		self.pi = pi
		self.u = u
		self.sigma = sigma
		if self.pi and self.u and self.sigma:
			self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
														for i in range(self.k)])

	def train(self, obs):
		"""
		use observation vector to train the GMM.

		Parameters
		----------
		obs : ndarray
			Each row of the k by dim array is an observation vector. The
			columns are the features seen during each observation.
		"""
		if not obs.ndim == self.dim:
			raise ValueError("dimension of observation must equal to dim")
		if not len(obs) > (10 * self.k):
			raise ValueError("train data set is too small")
		#if the parameters are not defined, use kmeans2 to get initial value
		if not (self.pi and self.u and self.sigma):
			label = np.array([])
			while True:
				try:
					self.u, label = kmeans2(obs, self.k, 10, missing='raise')
				except ClusterError:
					print "catch a ClusterError and re-run kmeans2!"
				else:
					break
			self.pi = np.histogram(label, range(0, self.k+1), density=True)[0]
			self.sigma = np.empty((self.k, self.dim, self.dim))
			for i in range(self.k):
				self.sigma[i] = np.cov(obs[label==i].T, ddof=0)
		#EM algorithm
		#update gaussian components
		self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
											 		for i in range(self.k)])

	def predict(self, x):
		norm_pds = np.vstack([self.comp[i].pdf(x) for i in range(self.k)])
		comp_pds = np.dot(np.diag(self.pi), norm_pds)
		pds = np.sum(comp_pds, axis=0)
		return pds

	def draw(self):
		for i in range(self.k):
			x,y = multivariate_normal.rvs(self.u[i], self.sigma[i], 300).T
			plt.plot(x,y,'x')
		plt.show()

if __name__ == '__main__':
	pass




















