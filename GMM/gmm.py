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

	def __em(self, x):
		print "before em", self.u, self.pi
		N = len(x)
		#E step
		norm_pds = np.vstack([self.comp[i].pdf(x) for i in range(self.k)])
		comp_pds = self.pi.reshape(self.k, 1) * norm_pds
		pds = np.sum(comp_pds, axis=0)
		gamma = (1/pds).reshape(N, 1) * comp_pds.T
		#M step
		Nk = np.sum(gamma, axis=0)
		for i in range(self.k):
			self.u[i] = 1 / Nk[i] * np.sum(gamma[:,i].reshape(N,1) * x, axis=0)

			x_nomal = x - self.u[i]
			g_i = gamma[:,i].reshape(1, N)
			tmp = np.dot(g_i * x.T, x)
			print 1/ Nk[i]
			print tmp.shape
			self.sigma[i] = 1 / Nk[i] * tmp
		self.pi = Nk / N
		#evaluate the log likelihood
		print "after em", self.u, self.pi

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
			#update gaussian components
			self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
												 		for i in range(self.k)])
		#EM algorithm
		self.__em(obs)
		#update gaussian components
		self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
											 		for i in range(self.k)])



	def predict(self, x):
		norm_pds = np.vstack([self.comp[i].pdf(x) for i in range(self.k)])
		comp_pds = self.pi.reshape(self.k, 1) * norm_pds
		pds = np.sum(comp_pds, axis=0)
		return pds

	def draw(self):
		for i in range(self.k):
			x,y = multivariate_normal.rvs(self.u[i], self.sigma[i], 300).T
			plt.plot(x,y,'x')
		plt.show()

if __name__ == '__main__':
	pass




















