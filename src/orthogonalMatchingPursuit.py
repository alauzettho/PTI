###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Alauzet Thomas
# Novembre 2018
#
#
# OMP
#
###################################################################################################################
###################################################################################################################


import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit


#retourne l'argument max d'une liste
def argmax(L):
	max = np.max(np.abs(L))
	i = 0
	while np.abs(L[i]) != max:
		i = i + 1
	return(i)


# Transpose une liste
def transpose(V):
	S = []
	for i in V :
		S.append([i])
	return(np.array(S))


# Retourne les atomes du dictionnaires en temps que base ainsi que les composantes de x dans la base parcimonieuse
def orthogonalMatchingPursuit(x, D, nl, nc) :
	n				= 0
	z				= np.zeros(nc)
	error			= x
	listM			= []
	listAtomes		= []
	nombreAtomes	= nc
	errnorm			= 101

	for i in range(nombreAtomes) :
		listAtomes.append(D[:,i])

	while (n < nl and errnorm > 100) :
		J = []
		t = 2.5
		for j in range(nombreAtomes) :
			compute_value = np.dot(listAtomes[j], error) / np.linalg.norm(listAtomes[j])
			J.append(compute_value)

		J		= np.nan_to_num(J)
		argm	= argmax(J)
		dm		= listAtomes[argm]
		listM.append(argm)

		if (n == 0) :
			phi = np.array(transpose(dm))
		else :
			phi = np.concatenate((phi, transpose(dm)), axis = 1)

		phit	= np.transpose(phi)
		zz		= np.dot(np.linalg.pinv(phi), x)
		error	= x - np.dot(phi, zz)
		n		= n + 1
		errnorm = np.linalg.norm(error)

	z[listM]	= zz
	return(phi, z, n)


# Retourne la representation parcimonieuse d'une matrice en iterant l'omp sur chaque colonnes
def OMPX(X, D, nl, nc) :
	(a, b) = np.shape(X)
	alpha = []

	for i in range(b) :
		(x, y, z) = orthogonalMatchingPursuit(X[:,i], D, nl, nc)						
		alpha.append(transpose(y))
		
	alpha = np.concatenate(alpha, axis = 1)
	return(alpha)


# Retourne la representation parcimonieuse via la librairie scipy
def OMPX2(X, D, nl, nc) :
	if ((nl, nc) != np.shape(D)) :
		print("ERROR DIM !!")
		quit()

	omp = OrthogonalMatchingPursuit(normalize = True)
	omp.fit(D, X)
	alpha = omp.coef_.T
	return(alpha)