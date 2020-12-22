import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from math import log2, sqrt, pi
from sklearn.cluster import KMeans
from PIL import Image

image1="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/alfa2.bmp"
image2="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/beee2.bmp"
image3="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/cible2.bmp"
image4="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/promenade2.bmp"
image5="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/city2.bmp"
images=[image1,image2,image3,image4,image5]

m1=[1,1,1]
sig1=[1,1,1]
m2=[4,2,1]
sig2=[1,1,9]

def lit_image(nom_image):
    img = cv.imread(nom_image,0)
    m, n = img.shape
    return img, m, n

def affiche_image(mat_image,titre):
    cv.imshow(titre,mat_image)
    cv.waitKey()
    #plt.imshow(mat_image, cmap='gray')
    #plt.title(titre)

def identif_classes(X):
    classes=[0,0]
    trouve=False
    hist = cv.calcHist([X],[0],None,[256],[0,256])
    for idi, i in enumerate(hist):
        if i!=0 and trouve==False:
            classes[0]=idi
            trouve=True
        elif i!=0 and trouve==True:
            classes[1]=idi
            return classes

def bruit_gauss(signal, m1, sig1, m2, sig2):
    classes=identif_classes(signal)
    signal_noisy = (signal == classes[0]) * np.random.normal(m1, sig1**0.5, signal.shape) + \
                   (signal == classes[1]) * np.random.normal(m2, sig2**0.5, signal.shape)
    return signal_noisy


# for i in range(5):
#     for j in range(3):
#         X, m, n =lit_image(images[i])
#         #affiche_image(X,'image')
#         signal_bruite=bruit_gauss(X,m1[j], sig1[j], m2[j], sig2[j])
#         #affiche_image(signal_bruite, 'image bruité')
#         plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/'+'image'+ str(i)+'_bruite'+ str(j)+'.png',signal_bruite,cmap='gray')

# #
# #

def gauss(signal_noisy, m1, sig1, m2, sig2):
    """
    Cette fonction transforme le signal bruitée en appliquant à celui-ci deux densitées gausiennes
    :param signal_noisy: Le signal bruité (numpy array 1D)
    :param m1: La moyenne de la première gaussienne
    :param sig1: L'écart type de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: L'écart type de la deuxième gaussienne
    :return: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément de signal noisy
    """
    gauss1 = (1 / (sig1 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((signal_noisy - m1) / sig1) ** 2))
    gauss2 = (1 / (sig2 * sqrt(2 * pi))) * np.exp(-(1 / 2) * (((signal_noisy - m2) / sig2) ** 2))
    return np.stack((gauss1, gauss2), axis=1)

def taux_erreur(A,B,m,n):
    return len(np.transpose(np.nonzero(A-B)))/(m*n)

def Kmean_classes(signal, cl1, cl2):
    m,n=signal.shape
    img = signal.reshape(-1,1)
    kmeans = KMeans(n_clusters=2).fit(img)
    cluster_centers=kmeans.cluster_centers_
    cluster_labels=kmeans.labels_
    X_reconstruit=cluster_centers[cluster_labels].reshape(n,m)
    for k in range(m):
        for i in range(n):
            if X_reconstruit.astype(int)[k][i]==cl1:
                X_reconstruit[k][i]=cl1
            else :
                X_reconstruit[k][i]=cl2
    #affiche_image(X_reconstruit,"ima")
    print(taux_erreur(X,X_reconstruit,m,n))
    return X_reconstruit

# X, m, n =lit_image(images[0])
#
# signal_bruite=bruit_gauss(X,m1[0], sig1[0], m2[0], sig2[0])
#
# Kmean_classes(signal_bruite,0,255)

# for i in range(5):
#      for j in range(1):
#         X, m, n =lit_image(images[i])
#         #affiche_image(X,'image')
#         signal_bruite=bruit_gauss(X,m1[j], sig1[j], m2[j], sig2[j])
#         #affiche_image(signal_bruite, 'image bruité')
#         X_recons=Kmean_classes(signal_bruite,0,255)
#         taux_erreur(X,X_recons,m,n)
#         plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/'+'image'+ str(i)+'_reconstruite'+ str(j)+'.png',X_recons,cmap='gray')

#
#
def calc_probaprio(X,m,n,cl1,cl2):
    signal=X.reshape(-1,1)
    p1 = np.sum((signal == cl1)) / signal.shape[0]
    p2 = np.sum((signal == cl2)) / signal.shape[0]
    return np.array([p1, p2])

def mpm_gauss(signal_bruite, cl1, cl2, p, m1, sig1, m2, sig2):
    w=np.array([cl1,cl2])
    signal_noisy=signal_bruite.reshape(-1,1)
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    proba_apost = p * gausses
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    X_seg=w[np.argmax(proba_apost, axis=1)]
    return X_seg[:,0].reshape(256,256)
#
#
# p=calc_probaprio(X, m, n, 0 , 255)
# X_reconstruit=mpm_gauss(signal_bruite,0,255,p,1,1,4,1)
# print(taux_erreur(X,X_reconstruit,m,n))
# affiche_image(X_reconstruit.astype(np.uint8),'test')

for i in range(5):
     for j in range(3):
        X, m, n =lit_image(images[i])
        signal_bruite=bruit_gauss(X,m1[j], sig1[j], m2[j], sig2[j])
        p=calc_probaprio(X, m, n, 0 , 255)
        X_recons=mpm_gauss(signal_bruite,0,255,p,1,1,4,1)
        print(taux_erreur(X,X_recons,m,n))
        plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/'+'image'+ str(i)+'_reconstruitempm'+ str(j)+'.png',X_recons,cmap='gray')