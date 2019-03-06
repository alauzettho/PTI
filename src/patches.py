###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Caissa Mathieu, Guichemerre Alexis
# Janvier 2018
#
#
# Patches
#
###################################################################################################################
###################################################################################################################


import numpy as np
import random
import operator
from PIL.Image import *


def distance(x, y) :
	xa = np.array([x[0], x[1]])
	ya = np.array([y[0], y[1]])
	return(np.linalg.norm(xa - ya))


def patchCoord(centre, taille) :
	x = centre[0]
	y = centre[1]
	res = []
	moitT = taille // 2

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
	img = imgName
	l,h = np.shape(imgName)
	s = taille + taille
	centers = []
	patch = []
	r0X = random.randint(taille, l - taille)
	r0Y = random.randint(taille, h - taille)
	centers.append((r0X,r0Y))
	cpt = n - 1
	for i in range(cpt):
		testD = True
		rxy = (random.randint(taille, l - taille), random.randint(taille, h - taille))
		for j in centers:
			dist=distance(j,rxy)
			if dist<taille :
				testD=False
		if testD:
			centers.append(rxy)
	for c in centers:
		coord = patchCoord(c,taille)
		xyMin = (c[0] - taille // 2, c[1] - taille // 2)
		tabPixel = []
		for coo in coord:
			tabPixel.append( img[coo[0]][coo[1]] )
		patch.append(( normalisationPatch(tabPixel),variancePatch(tabPixel), xyMin))
		patch.sort(key=operator.itemgetter(1), reverse=True)
	return patch


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


def imageToSubImageVectorized(imgName, nRow, nCol):
	lenImg	= np.shape(imgName)
	res		= []
	nbR		= int(lenImg[0] / nRow)
	nbC		= int(lenImg[1] / nCol)
	for k in range(nRow):
		for l in range(nCol):
			subI = []
			for i in range(nbR):
				for j in range(nbC):
					subI.append(imgName[i + k * nbR][j + l * nbC])
			res.append(subI)
	return np.array(res)


def vectorToImage(arr, nRow, nCol):
	nbR, nbC = np.shape(arr)
	lenC = len(arr[:,0])
	sqrtL = int(np.sqrt(lenC))
	res = []
	for k in range(nbC):	
		colon = arr[:,k]
		subA = np.zeros((sqrtL, sqrtL))
		for i in range(sqrtL):
			for j in range(sqrtL):
				subA[i][j] = colon[i*sqrtL+j]
		res.append(subA)

	imgRes = np.empty((nRow*sqrtL,nCol*sqrtL))
	imgL = np.empty((sqrtL,nCol*sqrtL))
	cpt = 0
	for a in res:
		if(cpt%nCol != 0):
			np.concatenate((imgL,a), axis = 1)
		else:
			np.concatenate((imgRes,imgL), axis = 0)
			imgL = np.empty((sqrtL,nCol*sqrtL))
		cpt += 1
	return imgRes