###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Alauzet Thomas
# Novembre 2018
#
#
# KSVD
#
###################################################################################################################
###################################################################################################################


import numpy as np


# Transpose une liste
def transpose(V) :
	S = []
	for i in V :
		S.append([i])

	return(np.array(S))


# Prend en entree un entier k, le vecteur initiale X, le dictionnaire D, une representation alpha, le nombre de ligne nl et retourne l'erreur Ek
def erreurk(k, X, D, alpha, nl) :
	Dk = D
	for i in range(nl) :
		Dk[i][k] = 0

	return(X - np.dot(Dk, alpha))


# Renvoit le support d'un vecteur alpha
def support(alpha) :
	L = []
	for i in range(len(alpha)) :
		if (alpha[i] != 0):
			L.append(i)

	return(L)


# Prend en entre un vecteur alpha et retourne la matrice de passage permettant de donner alpha sans zeros
def omega(alpha) :
	l		= np.size(alpha)
	K		= support(alpha)
	b		= len(K)
	omega	= np.zeros([l,b])

	for i in range(b) :
		omega[K[i]][i] = 1

	return(omega)


# Fait une iteration de la mise a jour du dictionnaire via la methode ksvd
# Alpha est une representation partimonieuse de X
# L'algorithme retourne le dictionnaire D et le nouveau vecteur partimonieu alpha
def ksvd(X, D, alpha):
	print("##################################################################################")
	(nl, nc)   = np.shape(D)
	(pix, piy) = np.shape(X)
	D2, alpha2 = [], []

	for k in range(nc):
		Ek		= erreurk(k, X, D, alpha, nl)
		omegak	= omega(alpha[k,:])

		# Support vide:
		if (np.shape(omegak)[1] < piy) :
			D2.append(transpose(np.random.rand(nl)))
			alpha2.append(transpose(X[0])) # np.random.randint(low = 0, high = pix)

		# Support non vide:
		else :	
			Ekr		  = np.dot(Ek, omegak)
			(U, M, V) = np.linalg.svd(Ekr, full_matrices = False)
			D2.append(transpose(U[:,0]))
			alpha2.append(transpose(M[0] * V[0,:]))

	alpha2	= np.concatenate(alpha2, axis = 1)
	D2		= np.concatenate(D2, axis = 1)
	return(D2, alpha2.T)