###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Alauzet Thomas, Caissa Mathieu, Guichemerre Alexis
# Novembre 2018
#
#
# Le but de ce programme est de supprimer des incoherences dans une image issue d'un dataset.
# Pour cela nous nous proposons de passe dans une base d'apprentissage afin de modifier l'information de l'image.
# L'apprentissage du dictionnaire se fait en partant d'une DCT.
# Le dictionnaire est mis a jour a chaque iteration via un algorithme de poursuite.
# L'image (signal) est alors transformee dans la base du dictionnaire.
# Dans cette base les incoherences disparaissent et en la retransformant, l'image devient plus net.
#
#
# Il est possible de ne pas calculer de dictionnaire.
# Dans ce cas, le code prend en entree une image et retourne sa version optimisee sans passe par le calcul de D.
# Il faut cependant avoir un dictionnaire sous la main et l'importer. Les etapes 2 et 3 ne seront pas effectuees.
#
#
# 4 dictionnaires initiaux sont possibles :
#
#	- celui forme de la DCT d'un signal (ici une ou plusieurs images)
#	- celui forme par la concatenation de plusieurs patch d'une image
#	- celui forme par la concatenation de plusieurs patch de plusieurs images
#	- choisir un dictionnaire deja calcule
#
#
# Dossier/src/main.py
#             ksvd.py
#             patche.py
#             discreteCosineTransform.py
#             orthogonalMatchingPursuit.py
#        /out/..
#        /images/..
#
# Dependances : OpenCV, numpy, ...
#
#
# Pour lancer le code :			python main.py
# Pour faire du profiling :		python -m cProfile main.py > profile.txt
# Pour specifier les param :	python main.py piy nc
#
#
# Sur Cesga :
#	- module load gcccore/7.3.0-glibc-2.25 python/3.7.0
#	- pip3 install numpy Pillow --user
#
###################################################################################################################
###################################################################################################################



################################################ Step 0 : Parametres ##############################################

main_folder = "/home/eisti/Documents/Ing3/PTI/"	# Path principale
images_path	= main_folder + "/images/"			# Path de nos images
output_path = main_folder + "/out/"				# Chemin des fichiers de sorti
dic_file	= output_path + "/dic.npy"			# Chemin vers notre dictionnaire en binaire
init_img	= "2007062308_cam01.jpg"			# Image initiale a choisir pour les method 0 et 1
opt_img		= "2007071912_cam01.jpg"			# Image a optimiser
out_img		= "/OutputImage.png"				# Nom de l'image de sortie
iterat_ksvd	= True								# Indique si nous devons calculer et sauvegarder un nouveau dictionnaire
init_method = 1									# Valeur entre 0 et 2 indiquant l'initialisation dit plus haute 
K			= 1									# Nombre d'iteration du ksvd





############################################ Step 1 : Initialisation ##############################################

import cv2
import os, sys
import numpy as np
import ksvd as ksvd
import patches as ptc
import discreteCosineTransform as dct
import orthogonalMatchingPursuit as omp


if len(sys.argv) > 1 :
	piy	= int(sys.argv[1])
else :
	piy	= 64								# Dimension de l'image selon l'absice, piyMax = 4224

if len(sys.argv) > 2 :
	nc	= int(sys.argv[2])
	if (nc < piy) :
		print("ERROR DIM !!")
		quit()
else :
	nc	= piy + 10							# Dimension du dictionnaire selon y, nc > piy

pix		= int(piy * 2376 / 4224)			# Dimension de l'image selon y, piyMax = 2376
nl		= pix								# Dimension du dictionnaire selon x, nl = pix
S		= int(np.sqrt(piy))					# Taille des patches d'apprentissage





###################################### Step 2 : Initialisation du Dictionnaire ####################################

print("################################## INITIALIZE ####################################")

X = cv2.imread(images_path + init_img)
X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
X = cv2.resize(X, (piy, pix))


# Import une image
if (init_method == 0) :
	INIT_DIC = cv2.imread(images_path + init_img)
	INIT_DIC = cv2.cvtColor(INIT_DIC, cv2.COLOR_BGR2GRAY)
	INIT_DIC = cv2.resize(INIT_DIC, (nc, nl))
	D 		 = dct.discreteCosineTransform(INIT_DIC)
	

# Import plusieurs patchs d'une image
elif (init_method == 1) :
	INIT_DIC = cv2.imread(images_path + init_img)
	INIT_DIC = cv2.cvtColor(INIT_DIC, cv2.COLOR_BGR2GRAY)
	D = ptc.start(INIT_DIC, nc, S).T
	D = D.copy()
	
	# ttt = int(np.sqrt(np.shape(D)[1]))
	# print(ttt)

	# PB = np.zeros([S, S * ttt])

	# for ll in range(ttt) :
	# 	PA = np.zeros((S, S))
	# 	for l in range(ttt) :
	# 		res = np.zeros((S, S))
	# 		for i in range(S):
	# 			for j in range(S):
	# 				res[i][j] = D[i * S + j][l + ttt * ll]
	# 		PA	= np.concatenate((PA, res), axis = 1)
	# 	PA = np.delete(PA, np.s_[0:S], axis=1)
	# 	PB = np.concatenate((PB, PA), axis=0)
	# 	print(np.shape(PB))
	# PB = np.delete(PB, np.s_[0:S], axis=0)
	# print(np.shape(PB))


	# print('end')
	# cv2.imwrite(output_path + "InitDicPatches.png", PB * 1000)

	D.resize((nl, nc), refcheck = False)


# Import plusieurs patchs de plusieurs image
elif (init_method == 2) :
	print("TODO")

	D = 0


# Import un dictionnaire existant
else :
	print("LOADING DICTIONARY")
	D = np.load(dic_file)



if ((nl, nc) != np.shape(D)) :
	print("ERROR DIM !!")
	quit()





############################################## Step 3 : Iteration KSVD ############################################

if (iterat_ksvd == True) :
	alpha = omp.OMPX(X, D, nl, nc)
	for i in range(K):
		print("################################# ITERATION : %s ##################################" %(i + 1,))
		
		D, alpha = ksvd.ksvd(X, D, alpha)
		alpha 	 = omp.OMPX(X, D, nl, nc)

		if ((nl, nc) != np.shape(D) or pix != nl or (nc, piy) != np.shape(alpha) or (pix, piy) != np.shape(X)) :
			print("ERROR DIM !!")
			quit()

		np.save(dic_file, D)
		print("Erreur apres la %s-ieme iteration : %s" %(i + 1, np.linalg.norm(X - np.dot(D, alpha)),))

	print("Fin de l'apprentissage de dictionnaire\n")


# On vient d'appliquer K fois ksvd, on retourne maintenant le dictionaire et la representation finale





############################################## Step 4 : Optimise les Images #######################################

print("############################### OPTIMIZATION #####################################")
from fractions import gcd

# Importe l'image a optimiser
OPT		= cv2.imread(images_path + opt_img)
OPT		= cv2.cvtColor(OPT, cv2.COLOR_BGR2GRAY)
OPT		= cv2.resize(OPT, (piy, pix))


print(np.shape(OPT))
# Selection des patches et vectorisation
a	= 
b	= 
c1	= gcd(a, 0)
c2	= 

OPT		= ptc.imageToSubImageVectorized(OPT, c1, c2)
print(np.shape(OPT))

print(np.shape(D))

# Applique OMP
alpha	= omp.OMPX(OPT, D, nl, nc)

# Devectorisation de l'image obtenue
#


# Reconstruction 
dot		= np.dot(D, alpha)

print("Erreur de reconstruction : %s" %(np.linalg.norm(X - dot)))

cv2.imwrite(output_path + out_img, dot)

# cv2.imwrite(output_path + "alpha.png", alpha)
# cv2.imwrite(output_path + "dic.png", D)
# print(np.count_nonzero(alpha))

print("################################## ALL DONE#######################################")