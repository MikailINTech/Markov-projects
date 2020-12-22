import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from math import log2, sqrt, pi
from sklearn.cluster import KMeans
from PIL import Image
from scipy.stats import norm
import random

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

def bruit_gauss(signal, m1, sig1, m2, sig2):
    classes=identif_classes(signal)
    signal_noisy = (signal == classes[0]) * np.random.normal(m1, sig1, signal.shape) + \
                   (signal == classes[1]) * np.random.normal(m2, sig2, signal.shape)
    return signal_noisy

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

## On oublie tout ce que l'on sait sur les paramètres
#p1 = p2 = m1 = m2 = sig1 =sig2 = 0


X, m, n =lit_image(images[2])
#affiche_image(X,'image')
signal_bruite=bruit_gauss(X,m1[0], sig1[0], m2[0], sig2[0])
#affiche_image(signal_bruite, 'image bruité')



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


def calc_probapost_gauss(Y,m,n,p1,p2,m1,sig1,m2,sig2):
    A= np.stack([p1*norm.pdf(Y,m1,sig1),p2*norm.pdf(Y,m2,sig2)],axis=2)
    A=A/A.sum(axis=2)[..., np.newaxis]
    return A


def calc_EM(Y,m,n,p10,p20,m10,sig10,m20,sig20,nb_iter):
    p=np.array([p10,p20])
    list_p1=[]
    list_p2=[]
    list_m1=[]
    list_m2=[]
    list_sig1=[]
    list_sig2=[]
    for k in range(nb_iter):
        if k==0:
            post_law=calc_probapost_gauss(Y,m,n,p10,p20,m10,sig10,m20,sig20)
        else :
            post_law=calc_probapost_gauss(Y,m,n,p1,p2,m1,sig1,m2,sig2)
        p=[np.sum(post_law[...,0])/Y.size,np.sum(post_law[...,1])/Y.size]
        m1 = np.sum(post_law[...,0]*Y) /np.sum(post_law[...,0])
        sig1 = np.sqrt((post_law[...,0]* ((Y- m1) ** 2)).sum() / (np.sum(post_law[...,0])-1))
        m2 = np.sum(post_law[...,1]*Y) /np.sum(post_law[...,1])
        sig2 = np.sqrt((post_law[...,1]* ((Y- m2) ** 2)).sum() / (np.sum(post_law[...,1])-1))
        p1=p[0]
        p2=p[1]
        list_p1.append(p1)
        list_p2.append(p2)
        list_m1.append(m1)
        list_m2.append(m2)
        list_sig1.append(sig1)
        list_sig2.append(sig2)
    k=[i for i in range(nb_iter)]

    plt.plot(k,list_p1,label="p1")
    plt.plot(k,list_p2,label='p2')
    plt.plot(k,list_m1,label='m1')
    plt.plot(k,list_sig1,label='sig1')
    plt.plot(k,list_m2,label='m2')
    plt.plot(k,list_sig2,label='sig2')
    plt.xlim(0,nb_iter)
    plt.ylim(0,10)
    plt.legend()
    plt.show()
    return p1,p2, m1, sig1, m2, sig2

# classes=identif_classes(X)
# prob=calc_probaprio(X,m,n,classes[0],classes[1])
# calc_EM(signal_bruite,m,n,prob[0],prob[1],m1[2]+12,sig1[2]+6,m2[2]+13,sig2[2]-1,50)

def est_empirique(X,Y,cl1,cl2):
    m,n=X.shape
    p=calc_probaprio(X,m,n,cl1,cl2)
    m1=np.sum(Y*(X==cl1))/(Y.size*p[0])
    m2=np.sum(Y*(X==cl2))/(Y.size*p[1])
    sig1 = np.sqrt(((X==cl1)* ((Y- m1) ** 2)).sum() / (Y.size*p[0]-1))
    sig2 = np.sqrt(((X==cl2)* ((Y- m2) ** 2)).sum() / (Y.size*p[1]-1))
    return p[0],p[1],m1,sig1,m2,sig2

def init_param(Y,cl1,cl2,iter):
    m,n= Y.shape
    X_kmeans=Kmean_classes(Y,cl1,cl2)
    p0,p1,m1,sig1,m2,sig2=est_empirique(X_kmeans,Y,cl1,cl2)
    return calc_EM(Y,m,n,p0,p1,m1,sig1,m2,sig2,iter)

def init_param_SEM(Y,cl1,cl2,iter):
    m,n= Y.shape
    X_kmeans=Kmean_classes(Y,cl1,cl2)
    p0,p1,m1,sig1,m2,sig2=est_empirique(X_kmeans,Y,cl1,cl2)
    return calc_SEM(Y,m,n,p0,p1,m1,sig1,m2,sig2,iter)

# for j in range(3):
#         signal_bruite=bruit_gauss(X,m1[j], sig1[j], m2[j], sig2[j])
#         p1r,p2r, m1r, sig1r, m2r, sig2r=init_param(signal_bruite,0,255,50)
#         X_mpm=mpm_gauss(signal_bruite,0,255,np.array([p1r,p2r]),m1r, sig1r, m2r, sig2r)
#         print(taux_erreur(X,X_mpm,m,n))
#         plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP2/image_bruit'+ str(j)+'.png',X_mpm,cmap='gray')

# p1r,p2r, m1r, sig1r, m2r, sig2r=init_param(signal_bruite,0,255,50)
# X_mpm=mpm_gauss(signal_bruite,0,255,np.array([p1r,p2r]),m1r, sig1r, m2r, sig2r)
# print(taux_erreur(X,X_mpm,m,n))
# affiche_image(X_mpm.astype(np.uint8),'test')

def tirage_apost(Ppost,cl1,cl2,m,n):
    A=np.random.rand(m,n)
    X_post=(A>Ppost[...,0])*cl2+(A<Ppost[...,0])*cl1
    return X_post

def calc_SEM(Y,m,n,p10,p20,m10,sig10,m20,sig20,nb_iter):
    p=np.array([p10,p20])
    list_p1=[]
    list_p2=[]
    list_m1=[]
    list_m2=[]
    list_sig1=[]
    list_sig2=[]
    for k in range(nb_iter):
        if k==0:
            post_law=calc_probapost_gauss(Y,m,n,p10,p20,m10,sig10,m20,sig20)
            X_post=tirage_apost(post_law,0,255,m,n)
        else :
            post_law=calc_probapost_gauss(Y,m,n,p1,p2,m1,sig1,m2,sig2)
            X_post=tirage_apost(post_law,0,255,m,n)
        p1,p2,m1,sig1,m2,sig2=est_empirique(X_post,Y,0,255)
        list_p1.append(p1)
        list_p2.append(p2)
        list_m1.append(m1)
        list_m2.append(m2)
        list_sig1.append(sig1)
        list_sig2.append(sig2)
    k=[i for i in range(nb_iter)]
    plt.plot(k,list_p1,label="p1")
    plt.plot(k,list_p2,label='p2')
    plt.plot(k,list_m1,label='m1')
    plt.plot(k,list_sig1,label='sig1')
    plt.plot(k,list_m2,label='m2')
    plt.plot(k,list_sig2,label='sig2')
    plt.xlim(0,10)
    plt.ylim(0,5)
    plt.legend()
    plt.show()
    return p1,p2, m1, sig1, m2, sig2


# p1r,p2r, m1r, sig1r, m2r, sig2r=init_param_SEM(signal_bruite,0,255,20)
# X_mpm=mpm_gauss(signal_bruite,0,255,np.array([p1r,p2r]),m1r, sig1r, m2r, sig2r)
# print(taux_erreur(X,X_mpm,m,n))

for j in range(3):
        signal_bruite=bruit_gauss(X,m1[j], sig1[j], m2[j], sig2[j])
        p1r,p2r, m1r, sig1r, m2r, sig2r=init_param_SEM(signal_bruite,0,255,50)
        X_mpm=mpm_gauss(signal_bruite,0,255,np.array([p1r,p2r]),m1r, sig1r, m2r, sig2r)
        print(taux_erreur(X,X_mpm,m,n))
        plt.imsave('C:/Users/duzen/OneDrive/Bureau/Cours TSP 2018-2019 (S2)/MSA/TPs Markov/TP2/image_bruit_SEM'+ str(j)+'.png',X_mpm,cmap='gray')