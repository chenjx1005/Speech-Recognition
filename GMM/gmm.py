#!usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from scipy.cluster.vq import kmeans2, ClusterError
from scipy.stats import multivariate_normal
import scipy.io


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
		# TODO: varify the parameters pi, u, sigma
		self.pi = pi
		self.u = u
		self.sigma = sigma
		if not (self.pi is None or self.u is None or self.sigma is None):
			self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
								  for i in range(self.k)])

	def __str__(self):
		info = "this GMM contains %d Gaussian Models.\n" % self.k
		m = ["\nModel %d: prior probability is %.3f\nmean value is %s\ncovariance" \
			 "is\n%s\n" % (i, self.pi[i], self.u[i], self.sigma[i]) \
			 for i in range(self.k)]
		return info + "".join(m)

	def __em(self, x):
		N = len(x)
		threshold = 0.001
		while True:
			# E step
			norm_pds = np.vstack([self.comp[i].pdf(x) for i in range(self.k)])
			comp_pds = self.pi.reshape(self.k, 1) * norm_pds
			pds = np.sum(comp_pds, axis=0)
			gamma = (1. / pds).reshape(N, 1) * comp_pds.T

			# M step
			u_old = np.copy(self.u)
			sigma_old = np.copy(self.sigma)
			pi_old = np.copy(self.pi)
			ln_likelihood_old = np.sum(np.log(self.predict(x)))

			Nk = np.sum(gamma, axis=0)
			for i in range(self.k):
				self.u[i] = 1. / Nk[i] * np.sum(gamma[:, i].reshape(N, 1) * x, axis=0)

				x_nomal = x - self.u[i]
				g_i = gamma[:, i].reshape(1, N)
				self.sigma[i] = np.dot(g_i * x_nomal.T, x_nomal) / Nk[i]
			
			self.pi = Nk / N
			self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
								  for i in range(self.k)])
			#evaluate the log likelihood and check for convergence
			ln_likelihood = np.sum(np.log(self.predict(x)))
			print "ln(likelihood) is ", ln_likelihood
			#TODO: set threshold according to data
			if ln_likelihood - ln_likelihood_old < threshold and \
					(np.fabs(self.u - u_old) < threshold).all() and \
					(np.fabs(self.sigma - sigma_old) < threshold).all() and \
					(np.fabs(self.pi - pi_old) < threshold).all():
				break

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
		# if the parameters are not defined, use kmeans2 to get initial value
		if self.pi is None or self.u is None or self.sigma is None:
			label = np.array([])
			while True:
				try:
					self.u, label = kmeans2(obs, self.k, 10, missing='raise')
				except ClusterError:
					print "catch a ClusterError and re-run kmeans2!"
				else:
					break
			print self.u
			self.pi = np.histogram(label, range(0, self.k + 1), density=True)[0]
			self.sigma = np.empty((self.k, self.dim, self.dim))
			for i in range(self.k):
				self.sigma[i] = np.cov(obs[label == i].T, ddof=0)
			# update gaussian components
			self.comp = np.array([multivariate_normal(self.u[i], self.sigma[i]) \
								  for i in range(self.k)])
		ln_likelihood = np.sum(np.log(self.predict(obs)))
		print "ln(likelihood) is ", ln_likelihood
		# EM algorithm
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
			plt.plot(self.u[:, 0], self.u[:, 1], 'ro')
			x, y = multivariate_normal.rvs(self.u[i], self.sigma[i], 300).T
			plt.plot(x, y, 'x')
		plt.show()


if __name__ == '__main__':
	pass




















