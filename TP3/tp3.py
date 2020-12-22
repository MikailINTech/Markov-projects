import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from math import log2, sqrt, pi
from sklearn.cluster import KMeans
from PIL import Image
from scipy.stats import norm
import random
import os ,sys
sys.path.append(os.path.abspath('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3'))
from tp3_tools import *

image1="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/alfa2.bmp"
image2="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/beee2.bmp"
image3="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/cible2.bmp"
image4="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/promenade2.bmp"
image5="C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP1/city2.bmp"
images=[image1,image2,image3,image4,image5]

m1=[1,1,1]
sig1=[1,1,1]
m2=[4,2,1]
sig2=[1,1,3]

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

def bruit_gauss(signal, m1, sig1, m2, sig2,classes):
    #classes=identif_classes(signal)
    signal_noisy = (signal == classes[0]) * np.random.normal(m1, sig1, signal.shape) + \
                   (signal == classes[1]) * np.random.normal(m2, sig2, signal.shape)
    return signal_noisy

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
    return X_reconstruit


def calc_probaprio(X,m,n,cl1,cl2):
    signal=X.reshape(-1,1)
    p1 = np.sum((signal == cl1)) / signal.shape[0]
    p2 = np.sum((signal == cl2)) / signal.shape[0]
    return np.array([p1, p2])

def mpm_gauss(signal_bruite,m,n,cl1, cl2, p, m1, sig1, m2, sig2):
    w=np.array([cl1,cl2])
    signal_noisy=signal_bruite.reshape(-1,1)
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    proba_apost = p * gausses
    proba_apost = proba_apost / (proba_apost.sum(axis=1)[..., np.newaxis])
    X_seg=w[np.argmax(proba_apost, axis=1)]
    return X_seg[:,0].reshape(m,n)


def calc_probapost_gauss(Y,m,n,p1,p2,m1,sig1,m2,sig2):
    A= np.stack([p1*norm.pdf(Y,m1,sig1),p2*norm.pdf(Y,m2,sig2)],axis=2)
    A=A/A.sum(axis=2)[..., np.newaxis]
    return A

def est_empirique(X,Y,cl1,cl2):
    m,n=X.shape
    p=calc_probaprio(X,m,n,cl1,cl2)
    m1=np.sum(Y*(X==cl1))/(Y.size*p[0])
    m2=np.sum(Y*(X==cl2))/(Y.size*p[1])
    sig1 = np.sqrt(((X==cl1)* ((Y- m1) ** 2)).sum() / (Y.size*p[0]-1))
    sig2 = np.sqrt(((X==cl2)* ((Y- m2) ** 2)).sum() / (Y.size*p[1]-1))
    return p[0],p[1],m1,sig1,m2,sig2

taux=[]
alpha=1
proba=calc_proba_champ(alpha)
classes=[0,255]
X_gibbs=gene_Gibbs_proba(65,65,classes,proba,10)
plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3/champ.png',X_gibbs,cmap='gray')
# for j in range(3):
#     #affiche_image(X_gibbs.astype(np.uint8),'test')
#     X_bruite=bruit_gauss(X_gibbs,m1[j],sig1[j],m2[j],sig2[j],classes)
#     plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3/champ_bruite'+ str(j) +'.png',X_bruite,cmap='gray')
#     X_bis=nouvelle_image(X_bruite)
#     X_mpm=MPM_proba_gauss(X_bis,classes,m1[j],sig1[j],m2[j],sig2[j],proba,10,2)
#     #affiche_image(X_mpm.astype(np.uint8),'test2')
#     X_estime=redecoupe_image(X_mpm)
#     plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3/champ_recons'+ str(j)+'.png',X_estime,cmap='gray')
#     taux.append(taux_erreur(X_gibbs,X_estime,65,65))



def init_param_EM(Y,m,n):
    cl1,cl2=0,255
    alpha=1
    p=calc_proba_champ(alpha)
    X=Kmean_classes(Y,cl1,cl2)
    m1=np.sum(Y*(X==cl1))/(np.sum(X==cl1))
    m2=np.sum(Y*(X==cl2))/(np.sum(X==cl2))
    sig1 = np.sqrt(((X==cl1)* ((Y- m1) ** 2)).sum() / (np.sum(X==cl1)))
    sig2 = np.sqrt(((X==cl2)* ((Y- m2) ** 2)).sum() / (np.sum(X==cl2)))
    return p,m1,sig1,m2,sig2




def EM_gibbsien_Gauss(Y,m,n,cl1,cl2,m1,sig1,m2,sig2,proba,nb_iter,nb_simu):
    X_simu=genere_Gibbs_proba_apost(Y,m1,sig1,m2,sig2,classes,proba,nb_iter)
    N=calc_N_part(X_simu,[cl1,cl2])
    proba=N
    N_post=calc_N_post(X_simu,[cl1,cl2])
    Ppost=N_post
    for i in range(1,nb_simu):
        X_simu=genere_Gibbs_proba_apost(Y,m1,sig1,m2,sig2,classes,proba,nb_iter)
        N=calc_N_part(X_simu,[cl1,cl2])
        proba += N
        N_post=calc_N_post(X_simu,[cl1,cl2])
        Ppost+=N_post
    proba=proba/nb_simu
    Ppost=Ppost/nb_simu
    for i in range(5):
        proba[i]=proba[i]/sum(proba[i])
    return proba , Ppost

def estim_param_bruit_gauss_EM(Y,m,n,classe,Ppost):
    post_law=Ppost
    m1 = np.sum(post_law[...,0]*Y) /np.sum(post_law[...,0])
    sig1 = np.sqrt((post_law[...,0]* ((Y- m1) ** 2)).sum() / (np.sum(post_law[...,0])))
    m2 = np.sum(post_law[...,1]*Y) /np.sum(post_law[...,1])
    sig2 = np.sqrt((post_law[...,1]* ((Y- m2) ** 2)).sum() / (np.sum(post_law[...,1])))
    return m1, sig1, m2, sig2


def EM_gauss(Y,m,n,cl1,cl2,nb_iter_EM,nb_iter,nb_simu):
    proba,m1,sig1,m2,sig2=init_param_EM(Y,m,n)
    print(proba,m1,sig1,m2,sig2)
    for i in range(nb_iter_EM):
        proba,Ppost=EM_gibbsien_Gauss(Y,m,n,cl1,cl2,m1,sig1,m2,sig2,proba,nb_iter,nb_simu)
        m1, sig1, m2, sig2 = estim_param_bruit_gauss_EM(Y,m,n,classes,Ppost)
    return proba, m1, sig1, m2, sig2


X_gibbs, m, n =lit_image(images[2])
for j in range(3):
    # affiche_image(X_gibbs.astype(np.uint8),'test')
    X_bruite=bruit_gauss(X_gibbs,m1[j],sig1[j],m2[j],sig2[j],classes)
    plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3/champ_bruite_a'+ str(j) +'.png',X_bruite,cmap='gray')
    probaf, m1f, sig1f, m2f, sig2f=EM_gauss(X_bruite,m,n,classes[0],classes[1],5,7,2)
    print(m1f, sig1f, m2f, sig2f)
    X_bis=nouvelle_image(X_bruite)
    X_mpm=MPM_proba_gauss(X_bis,classes,m1f, sig1f, m2f, sig2f,probaf,10,2)
    #affiche_image(X_mpm.astype(np.uint8),'test2')
    X_estime=redecoupe_image(X_mpm)
    plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP3/champ_recons_a'+ str(j)+'.png',X_estime,cmap='gray')
    taux.append(taux_erreur(X_gibbs,X_estime,m,n))

print(taux)
#print(proba)
#proba,Ppost= EM_gibbsien_Gauss(X_bruite,m,n,classes[0],classes[1],1,1,1,3,proba,1,1)
#print(proba)
#print(estim_param_bruit_gauss_EM(X_bruite,m,n,classes,Ppost))
#probaf, m1f, sig1f, m2f, sig2f=EM_gauss(X_bruite,m,n,classes[0],classes[1],1,1,1)
#print(probaf, m1f, sig1f, m2f, sig2f)