import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from math import log2, sqrt, pi
from sklearn.cluster import KMeans
from PIL import Image
from scipy.stats import norm
import random

def calc_proba_champ(alpha, normalization=True):
    '''
    crée un np array de taille 5x2 avec
    proba[i, j] = p(x_s=j|"il y a i voisins de type j")
    '''
    proba = np.empty((5, 2))
    for i in range(5):
        proba[i, 0] = np.exp((2 * i - 4) * alpha)
        proba[i, 1] = np.exp((2 * (4 - i) - 4) * alpha)
        if normalization:
            proba[i] /= np.sum(proba[i])
    return proba

def gene_Gibbs_proba(mm, nn, classe, proba, nb_iter):
    '''
    génère un champ de Markov de taille mmxnn
    de classes et lois locales données à l'aide de
    l'échantillonneur de Gibbs
    '''
    # initialisation aléatoire de l'échantillonneur de Gibbs
    X = np.random.randint(0, 2, size=(mm, nn))
    X= (X==0)*classe[0] +(X==1)*classe[1]

    table_voisins = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    # pour chaque itération de l'échantillonneur de Gibbs
    for k in range(nb_iter):
        print("Gibbs itération numéro", k)
        # pour chaque site
        for i in range(1, mm - 1):
            for j in range(1, nn - 1):
                # on récupère la configuration du voisinage
                # i.e. on compte le nombre de voisins à la classe 1
                config = 0
                for v in table_voisins:
                    config += (X[i + v[0], j + v[1]] == classe[1])
                distribution_locale = proba[config]
                # on met à jour le site avec un tirage selon la
                # distribution locale
                u = np.random.rand(1, 1)
                X[i, j] = ((u <= distribution_locale[1]) * classe[0] +
                    (u > distribution_locale[1]) * classe[1])

    return X

def redecoupe_image(X_in):
    '''
    enlève les bords de l'image qui ne peuvent pas être mis à jour avec
    l'échantillonneur de Gibbs
    '''
    return X_in[1:X_in.shape[0] - 1, 1:X_in.shape[1] - 1]

def nouvelle_image(Y_in):
    '''
    Ajoute une nouvelle rangée de pixel autour de l'image bruitée pour pouvoir
    faire tourner l'échantillonneur de Gibbs sur tous les pixels d'intérêt
    '''
    return np.pad(Y_in, ((1, 1), (1, 1)), mode="edge")

def MPM_proba_gauss(Y, classe, m1, sig1, m2, sig2, proba, nb_iter,
    nb_simu, init_gibbs=None):
    '''
    Algorithme du calcul du MPM de Marroquin pour un champ de Markov caché à
    bruit gaussien indépendant
    '''
    X_MPM_stacked = np.zeros((Y.shape[0], Y.shape[1], nb_simu))
    # pour toutes les différentes simulations MPM
    for n in range(nb_simu):
        print("MPM itération numéro", n)
        X_MPM_stacked[..., n] = genere_Gibbs_proba_apost(Y, m1, sig1, m2, sig2,
        classe, proba, nb_iter, init_gibbs)

    # choix en chaque site de la classe la plus tirée lors dessimulations MPM
    _cl0 = np.sum(X_MPM_stacked == classe[0], axis=-1)
    X_MPM = ((_cl0 >= nb_simu // 2) * classe[0] + (_cl0 < nb_simu // 2) *
    classe[1])
    return X_MPM

def genere_Gibbs_proba_apost(Y, m1, sig1, m2, sig2, classe, proba, nb_iter,
    init_gibbs=None):
    '''
    Échantillonneur de Gibbs pour le tirage a posteriori 'un champ de Markov
    caché à bruit gaussien indépendant
    '''
    if init_gibbs is None:
        # initialisation aléatoire de l'échantillonneur de Gibbs
        init_gibbs = np.random.randint(0, 2, size=Y.shape)
        init_gibbs = (init_gibbs==0)*classe[0]+(init_gibbs==1)*classe[1]
    X = init_gibbs

    table_voisins = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    # pour chaque itération de l'échantillonneur de Gibbs
    for k in range(nb_iter):
        print("Gibbs a posteriori itération numéro", k)
        # pour chaque site
        for i in range(1, Y.shape[0] - 1):
            for j in range(1, Y.shape[1] - 1):
                # on récupère la configuration du voisinage
                # i.e. on compte le nombre de voisins à la classe 1
                config = 0
                for v in table_voisins:
                    config += (X[i + v[0], j + v[1]] == classe[0])
                distribution_locale_apost = proba[config].copy() # copy !
                distribution_locale_apost[0] *= (1 / (np.sqrt(2 * np.pi) * sig1)
                    * np.exp(-0.5 * (Y[i, j] - m1) ** 2 / sig1 ** 2))
                distribution_locale_apost[1] *= (1 / (np.sqrt(2 * np.pi) * sig2)
                    * np.exp(-0.5 * (Y[i, j] - m2) ** 2 / sig2 ** 2))
                distribution_locale_apost /= np.sum(distribution_locale_apost)

                # on met à jour le site avec un tirage selon la
                # distribution locale a posteriori
                u = np.random.rand(1, 1)
                X[i, j] = ((u <= distribution_locale_apost[0]) * classe[0] +
                    (u > distribution_locale_apost[0]) * classe[1])

    return X

def calc_N_part(X_simu, classe):
    '''
    Estimation des Nij apparaissant dans l'EM Gibbsien par une approche MCMC
    '''
    table_voisins = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    N = np.zeros((5, 2))
    for i in range(1, X_simu.shape[0] - 1):
        for j in range(1, X_simu.shape[1] - 1):
            config = 0
            for v in table_voisins:
                config += (X_simu[i + v[0], j + v[1]] == classe[0])
            if X_simu[i, j] == classe[0]:
                N[config, 0] += 1
            else:
                N[config, 1] += 1

    return N


def calc_N_post(X_simu, classe):
    '''
    Estimation des probabilités a posteriori en chaque site par une approche
    MCMC
    '''
    N_post = np.stack([(X_simu == classe[0]), (X_simu == classe[1])], axis=-1)
    return N_post


