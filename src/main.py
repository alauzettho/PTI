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
# 3 dictionnaires initiaux sont possibles :
#
#	- celui de la DCT d'un signal (ici une images)
#	- celui par la concatenation de plusieurs patch d'une image
#	- celui d'un choix de dictionnaire
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
# Dependances : OpenCV, Numpy
#
#
# Pour lancer le code :			python main.py
# Pour faire du profiling :		python -m cProfile main.py > profile.txt
# Pour specifier les param :	python main.py piy nc
#								avec piy la taille de l'image et nc le nombre de colonnes du dictionnaire
#
#
# Sur Cesga :
#	- module load gcccore/7.3.0-glibc-2.25 python/3.7.0
#	- pip3 install numpy Pillow --user
#
###################################################################################################################
###################################################################################################################


import cv2
import os, sys
import numpy as np
import ksvd as ksvd
import patches as ptc
import discreteCosineTransform as dct
import orthogonalMatchingPursuit as omp





################################################ Step 0 : Parametres ##############################################


main_folder = os.getcwd() + "/../"				# Path principale
images_path	= main_folder + "/images/"			# Path de nos images
output_path = main_folder + "/out/"				# Chemin des fichiers de sorti
dic_file	= output_path + "/dic.npy"			# Chemin vers notre dictionnaire en binaire
init_img	= "2007062308_cam01.jpg"			# Image initiale a choisir pour les method 0 et 1
opt_img		= "2007071912_cam01.jpg"			# Image a optimiser
out_img		= "/OutputImage.png"				# Nom de l'image de sortie
iterat_ksvd	= True								# Indique si nous devons calculer et sauvegarder un nouveau dictionnaire
init_method = 0									# Valeur entre 0 et 1 indiquant l'initialisation dite plus haut
K			= 1									# Nombre d'iteration du ksvd





############################################ Step 1 : Initialisation ##############################################

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


# Importe une image
if (init_method == 0) :
	INIT_DIC = cv2.imread(images_path + init_img)
	INIT_DIC = cv2.cvtColor(INIT_DIC, cv2.COLOR_BGR2GRAY)
	INIT_DIC = cv2.resize(INIT_DIC, (nc, nl))
	D 		 = dct.discreteCosineTransform(INIT_DIC)
	

# Importe plusieurs patchs d'une image
elif (init_method == 1) :
	INIT_DIC = cv2.imread(images_path + init_img)
	INIT_DIC = cv2.cvtColor(INIT_DIC, cv2.COLOR_BGR2GRAY)
	D = ptc.start(INIT_DIC, nc, S).T
	D = D.copy()
	D.resize((nl, nc), refcheck = False)


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

# Importe l'image a optimiser
OPT		= cv2.imread(images_path + opt_img)
OPT		= cv2.cvtColor(OPT, cv2.COLOR_BGR2GRAY)
OPT		= cv2.resize(OPT, (piy, pix))
alpha	= omp.OMPX(OPT, D, nl, nc)
dot		= np.dot(D, alpha)

print("Erreur de reconstruction : %s" %(np.linalg.norm(X - dot)))
cv2.imwrite(output_path + out_img, dot)

print("################################## ALL DONE#######################################"),