###################################################################################################################
###################################################################################################################
#
# Projet Traitement des Images
# Thomas Alauzet, Caissa Mathieu, Guichemerre Alexis
# Janvier 2018
#
#
# Lecture des images
#
###################################################################################################################
###################################################################################################################


import os, sys
from PIL.Image import *
from pylab import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Redimensionne une image en taille de pixels (x,y)
def redimensionneImage(x, y, namefile) :
	im	= open(namefile)
	out	= im.resize((x, y))
	out.save(namefile[:-4] + "_resize.jpg")


# Importe une image en format binaire
def convertImageGris(namefile) :
	jpgfile = open(namefile)
	(largeur, hauteur) = jpgfile.size

	res = zeros((hauteur, largeur), dtype = 'i')

	for x in range(hauteur) :
		for y in range(largeur) :
			pixel = jpgfile.getpixel((y,x))
			res[x][y] = (pixel[0] + pixel[1] + pixel[2]) / 3
	return res


# Importe l'image
def importation(pix, piy, namefile) :
	redimensionneImage(piy, pix, namefile)
	res = convertImageGris(namefile[:-4] + "_resize.jpg")
	os.remove(namefile[:-4] + "_resize.jpg")
	return(res)