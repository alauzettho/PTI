###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Alauzet Thomas, Mathieu Caissa, Alexis Guichemerre
# Novembre 2018
#
###################################################################################################################
###################################################################################################################


import random
import operator
import numpy as np


def distance(x, y) :
	return(np.linalg.norm(np.array([x[0], x[1]]) - np.array([y[0], y[1]])))


def patchCoord(centre, taille) :
	x		= centre[0]
	y		= centre[1]
	res		= []
	moitT	= taille // 2

	for i in range(-moitT, moitT) :
		for j in range (-moitT, moitT) :
			res.append((i + x, j + y))

	return(res)


def variancePatch(patch) :
	moy1 = 0
	moy2 = 0
	for pixel in patch :
		moy1 += int(pixel)
		moy2 += int(pixel) * int(pixel)
	moy1 /= len(patch)
	moy2 /= len(patch)
	return(moy2 - (moy1 * moy1))


def moyenPatch(patch) :
	moy = 0
	for pixel in patch :
		moy += pixel
	moy /= len(patch)
	return(moy)


def normalisationPatch(patch) :
	moy = moyenPatch(patch)
	var = variancePatch(patch)
	res = []
	if (var == 0) :
		var = 1
		moy = 0
	for pixel in patch :
		tank = (pixel - moy) / np.sqrt(var)
		res.append(tank)
	return(res)


def patcheCenter(imgName, n, taille):
	img		= imgName
	l, h	= np.shape(imgName)
	centers	= []
	patch	= []
	r0X		= random.randint(taille, l - taille)
	r0Y		= random.randint(taille, h - taille)
	cpt		= n - 1
	centers.append((r0X, r0Y))

	for i in range(cpt) :
		testD	= True
		rxy		= (random.randint(taille, l - taille), random.randint(taille, h - taille))
		for j in centers:
			if distance(j, rxy) < taille :
				testD = False
		if testD :
			centers.append(rxy)

	for c in centers :
		coord		= patchCoord(c,taille)
		xyMin		= (c[0] - taille // 2, c[1] - taille // 2)
		tabPixel	= []
		for coo in coord :
			tabPixel.append(img[coo[0]][coo[1]])

		patch.append((normalisationPatch(tabPixel),variancePatch(tabPixel), xyMin))
		patch.sort(key = operator.itemgetter(1), reverse = True)
	return(patch)


def getPatches(liste) :
	res = []
	s   = int(np.sqrt(len(liste[0][0])))

	for i in liste :
		tank = []
		for k in range(0, s) :
			ligne = []
			for kk in range(0, s) :
				ligne.append(i[0][k * s + kk])
			tank.append(ligne)
		res.append(np.array(tank))
	return(res)


def start(img, nbP, taille):
	A = getPatches(patcheCenter(img, nbP, taille))
	B = [A[x].flatten() for x in range(len(A))]
	return(np.array(B))