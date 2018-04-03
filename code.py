#!/usr/bin/env python
# -*- coding: utf-8 -*-

import email
import re
import matplotlib.pyplot as plt
import math, random
from pprint import pprint
import numpy as np
from collections import Counter
import random

NB_MOTS_FREQUENTS = 20 # dimensions du vecteur de mots utilisé dans la partir 3

## TOUTS LES TESTS EFFECTUE  ON ETE COMMENTE!!!!!
"""#################################################################################################################"""
 ## FONCTION DE TRAITEMENT DES FICHIERS

def read_file(fname):
    """ Lit un fichier compose d'une liste de emails, chacun separe par au moins 2 lignes vides."""
    f = open(fname,'rb')
    raw_file = f.read()
    f.close()
    raw_file = raw_file.replace(b'\r\n',b'\n')
    emails =raw_file.split(b"\n\n\nFrom")
    emails = [emails[0]]+ [b"From"+x for x in emails[1:] ]
    return emails

def get_body(em):
    """ Recupere le corps principal de l'email """
    body = em.get_payload()
    if type(body) == list:
        body = body[0].get_payload()
    try:
        res = str(body)
    except Exception:
        res=""
    return res

def clean_body(s):
    """ Enleve toutes les balises html et tous les caracteres qui ne sont pas des lettres """
    patbal = re.compile('<.*?>',flags = re.S)
    patspace = re.compile('\W+',flags = re.S)
    return re.sub(patspace,' ',re.sub(patbal,'',s))

def get_emails_from_file(f):
    mails = read_file(f)
    return [ s for s in [clean_body(get_body(email.message_from_bytes(x))) for x in mails] if s !=""]

spam = get_emails_from_file("spam.txt" )
nospam = get_emails_from_file("nospam.txt")

"""#####################################################################################################################"""
## exercice 1: fonction split qui divise uniformément une liste en deux sous-listes.

def split(liste, x):
	l1 = []
	l2 = []
	j = math.floor(len(liste)*x)
	
	l1 = liste[0:j]
	l2 = liste[j:]
	
	return l1, l2

#fonction qui retourne la longueur de l'email selon le nombre de mots qui le constituent
def lenEmail (email):
    return len(email.split())



#fonction qui retourne la longueur de l'email selon le nombre de characteres
def len_email_char(email):
    return len(email)



#retourne une liste de longueur d'un ensemble d'emails donné on peut ici implementer le fais qu'on prend des intervalles
def liste_par_longeur(lem):
	l=[]
	for i in lem:
		l.append(lenEmail(i))
		
	return l



"""###########################TEST##########################################"""
##on trace l'histogramme pour les emails spam selon la longueur on a pris <2000 pour observer les variations 
#l1=[]
#l1 = [len(email.split()) for email in spam if len(email.split())<2000]
#
#plt.hist(l1,bins=50)
#plt.show()


###on trace l'histogramme pour les emails non spam selon la longueur
#l2=[]   
#l2 = [len(email.split()) for email in nospam if len(email.split())<2000]
#
#plt.hist(l2,bins=100)
#plt.show()


    
"""#####################################################################################################"""
##2eme EXERCICE CLASSIFICATION PAR LONGUEUR DE L'EMAIL

#Fonction apprend modele qui permet de renvoyer la distribution voulue
#cette fonction apprend mail renvoie une liste ou chaque case coorespond a un intervalle de longeur et l'attribut affecté a cet intervalle
def apprend_modele(spamem,nonspamem,intervalle):
    	#on va consider distrubtion uniforme p(Y=+1) = 0.5
        # on lui passe en parametre la valeur de bins
    intervallei=[]
    total=[]
    cpt=100
    spam=liste_par_longeur(spamem)
    nonspam=liste_par_longeur(nonspamem)
    spam=sorted(spam)
    nonspam=sorted(nonspam)
    #on va regarder les variations jusqua 2000 et tout les mails > 2000 on va les classer dans une seule categorie
    while (cpt<=2000):
        nb_spam=0
        nb_non_spam=0
        intervallei.append(cpt)
        for i in spam:
            if (cpt<i<=cpt+intervalle):
                nb_spam=nb_spam+1
        for i in nonspam:
            if (cpt<i<=cpt+intervalle):
                nb_non_spam=nb_non_spam+1
        intervallei.append(cpt+intervalle)
        if ((float(nb_spam)/len(spam))>(float(nb_non_spam)/len(nonspam))):
                intervallei.append(-1)
        else:
                intervallei.append(+1)
        total.append(intervallei)
        cpt=cpt+intervalle
   
    return total
    
    
#Fonction predi emailqui permet de renvoyer le label d'un email a partir d un modele obtenue par apprend modele
def predict_email(email, modele):
    #le modele ici est la distribution des spam selon l'intervalle des longueur retourne par apprend_modele
	#renvoie la liste des labels pour l'ensemble des emails en fonction du modele passe en parametre
    for m in modele:
        if (lenEmail(email)<=m[1]):
            return m[2]


        
#cette fonction nous retourne l'accuracy sur un ensemble des emails de test labelles         
def accuracy(emails, modele):
    nb_ok=0
    for e in emails:
        if (predict_email(e[0],modele) == e[1]):
            nb_ok=nb_ok+1
    return nb_ok/float(len(emails))


#fonctions qui retourne la probabilité d'erreur
def proba_err(emails,modele):
	return (1.0-accuracy(emails,modele))   



#fonction qui ajoute le label +1 ou 61 au liste passees en parametres
def ajout_label(l1,l2):

    res=[]
    for e in l1:
        t=[]
        t.append(e)
        t.append(+1)
        res.append(t)
    for e in l2:
        t=[]
        t.append(e)
        t.append(-1)
        res.append(t)
        
    return res


"""#####################################TEST##############################################"""
#l1_s,l2_s=split(spam, 0.8)
#l1_ns,l2_ns=split(nospam, 0.8)
#c=apprend_modele(l1_s,l1_ns,50)
#l=ajout_label(l2_s,	l2_ns)
#print(proba_err(l,c))




"""########################################################################################################"""
## 3eme exercice Classification a partir du contenu(mots) d'un email


#Fonction qui  calcule la frequence de mot dans l'ensemble passé en paramértre
def mail_freq(emails,nb):    
    all_words = []
    words=[]
    reg=r"[0-9_@\\\/]+"       
    for mail in emails:    
        w = mail.split()
        for mot in w:
            if(re.match(reg,mot)is None and len(mot)<27 and len(mot)>3 and mot.lower() not in words):# on elimine les mots inutiles
                words.append(mot.lower()) #on transforme tout les mots en miniscules pour eviter les redoublons
        all_words += words
    
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(nb)
    random.shuffle(dictionary)
    res=[]
    for i in dictionary:
        res.append(list(i))
    return res
    
    
#cette fonction prend en parametre notre ensemble train et retourne l'ensemble des mots possibles (on prends 2000  mots plus fréquents, on change apres pour tester)
#apres avoir enlever les mots inutiles 
def make_dic_init(emails):
    t=mail_freq(emails)
    for i in t:
        i[1]=0    
    return t
    
    
    
#cette fonction trenasforme un email en un dictionnaire en fonctions de nos mots
def make_dic(email,mots):
    t=mots
    for i in t:
        if(i[0] in email.split()):
            i[1]=1    
    return t



## fonction apprend modele du classifieur bayesien 
def apprend_modele_m(words,e,s):
    res=[]
    for w in words:
        m=[]
        nb_spam=0
        nb_non_spam=0
        for i in s:
            if (w[0] in (i.lower().split())):
                nb_spam=nb_spam+1
        for j in e:
            if(w[0] in j.lower().split()):
                nb_non_spam=nb_non_spam+1
        m.append(w[0])
        m.append(nb_spam/len(s)) # P(Xi=xi|y=+1)
        m.append(nb_non_spam/len(e)) # P(Xi=xi|y=+1)
        res.append(m)
    
    return res


## fonction de prediction du classifieur bayesien 
def predict_emailm(email, modele):
    lamda=0.0000000000000000000000000000000001
    random.shuffle(modele)
    dic=make_dic(email,modele)
    ps=0
    pns=0
    for i in range(0,len(dic)):
        if dic[i][1]==0:
            ps=ps-modele[i][1]*lamda
            pns=pns-modele[i][1]*lamda
        else:
            ps=ps+(math.log2(modele[i][1]+lamda))
            pns=pns+(math.log2(modele[i][2]+lamda))
    if((math.exp(ps))>(math.exp(pns))):
        return +1
    else:
        return -1
        
        
##cette fonction nous retourne l'accuracy sur un ensemble des emails de test labelles           
def accuracym(emails, modele):
    nb_ok=0
    for e in emails:
        if (predict_emailm(e[0],modele) == e[1]):
            nb_ok=nb_ok+1
    return nb_ok/float(len(emails))


def proba_errm(emails,modele):
	return (1.0-accuracym(emails,modele))   
 
 
"""#####################################TEST##############################################""" 
#l=l1_ns+l1_s
#random.shuffle(l)
#d=make_dic_init(l)
#res=apprend_modele_m(d,l1_ns,l1_s)
#l=ajout_label(l2_s,l2_ns)
#print(proba_errm(l,res))

#on dessine un histogramme avec les frequences des mots
#l1_s,l2_s=split(spam, 0.5)
#l1_ns,l2_ns=split(nospam, 0.5)  
#l=l1_s+l1_s
#freq=mail_freq(l,5000)
#t=[]
#for i in freq:
#    t.append(i[1])
#plt.hist(t)
#plt.show()

#dic=[3,30,300,3000]
#prb_er2=[]
#for i in dic:
#    l1_s,l2_s=split(spam, 0.7)
#    l1_ns,l2_ns=split(nospam, 0.7)  
#    l=l1_s+l1_s
#    random.shuffle(l)
#    d=make_dic_init(l,i)
#    res=apprend_modele_m(d,l1_ns,l1_s)
#    l=ajout_label(l2_s,l2_ns)
#    prb_er2.append(proba_errm(l,res))
#print(prb_er2)
#plt.title("L'evaluation de  proba d'erreur selon la taille de dictionaire ")
#plt.plot(dic,prb_er2)
#plt.xlabel('Taille Dictionnaire')
#plt.ylabel('Proba erreur')
#plt.show()






"""#############################################################################################################"""
##2EME PARIE:VISUALISATION

#fonction qui calcule la distance entre deux vecteur
#def dist(xi,xj):
#    return -np.dot(xi,xj) / (np.linalg.norm(xi)*np.linalg.norm(xj))

def dist(xi,xj):
    somme = 0.0
    for i in range(len(xi)):
        somme = somme + ((xi[i]-xj[i])**2)    
    
    return math.sqrt(somme)



# a utilier dans l'algo sne 
def get_freq(email, mots):
    _freqs = mail_freq([email],30)
    result = {x: 0 for x in mots}

    for word, freq in _freqs:
        if word in result:
            result[word] = freq

    return list(result.values())





#fonction qui calcule la formule Pij    
def Pij(listeX, i, j, sigma):
    num   = math.exp((-dist( listeX[i], listeX[j])) / 2*sigma )
    denom = 0.0
    for k in range(len(listeX)):
        if k != i:
            denom += math.exp(-dist(listeX[k],listeX[i])/(2*sigma))
            
    return num/denom




#fonction qui calcule la formule Qij
def Qij(listeY, i, j):
    denom = 0.0
    num   = math.exp(-dist(listeY[i],listeY[j]))
    for k in range(len(listeY)):
        if k != i:
            denom += math.exp(-dist(listeY[k], listeY[i]))
    
    return num/denom if denom != 0 else 0




## fonction qui calcule la divergence de Kullback-Leibler  a partir de Pij et des Qij
def KL(listePij, listeQij, i):
    if len(listePij) != len(listeQij):
        #print 'listePij et listeQij n\'ont pas la meme longueur'
        return -1 #Erreur, les deux listes doivent avoir la meme taille, car elles sont calculees a partir du meme ensemble de mails
    
    somme = 0
    for j in range(len(listePij)):
        if listeQij[i][j] == 0 or listePij[i][j] == 0: 
            #On ignore quand Pij ou Qij == 0 
            #i.e. 2 mails qui n'ont aucun mot en commun !!
            #on fait ca car on peut pas faire np.log(0)
            continue
        
        somme += listePij[i][j] * np.log(listePij[i][j] / listeQij[i][j])

    return somme



## Fonction qui calcule la descente du gradient 
def gradient(listePij, listeQij, listeY, i):
    somme_y1, somme_y2 = 0, 0
    for j in range(len(listeY)):
        common    = listePij[i][j]-listeQij[i][j] + listePij[j][i]-listeQij[j][i]
        somme_y1 += common * ( listeY[i][0] - listeY[j][0] )
        somme_y2 += common * ( listeY[i][1] - listeY[j][1] )

    return 2*somme_y1, 2*somme_y2


def calcul_c(listePij, listeY):
    listeQij     = []
    for i in range(len(listeY)):
        listeQij.append([])
        for j in range(len(listeY)):
            listeQij[i].append( Qij(listeY, i, j) )

    C = 0
    for i in range(len(listePij)):
        C += KL(listePij, listeQij, i)

    return C, listeQij


#Implementation de l'algo SNE selon les etapes décrite dans le pseudo-code
def algo_sne():
    #la fonction de convergence test si la dernièere valeur de C calculé est inferieur à une certaine petite valeur (pas de différence)
    def sne_converge(historique_C): 
        if historique_C[-1] <= 0:
            return True
        return False

    ##Etape 0: Chargement des fichiers et des emails
    spam         = get_emails_from_file("spam.txt" )
    nospam       = get_emails_from_file("nospam.txt")

    l1_s,  l2_s  = split(spam,   0.02) # on teste sur un petit nombre de mails sinon prend beaucoup de temps
    l1_ns, l2_ns = split(nospam, 0.03)
    
    emails       = l1_ns + l1_s
    random.shuffle(emails) #on a mélangé un peu les emails entre spam et vrais mails
    print ('NB mail:', len(emails), 'dont', len(l1_s), 'sont spams.')
    #la liste des 1000 mots les plus fréquences sur toutes la base des emails
    mots         = [ mot[0] for mot in mail_freq(emails,30)]

    ##Etape 1 et 2: Calcul des Pij/Qij et generation des Xi et Yi
    
    #a. Liste des vecteurs Xi = list[ Xi=vect()   pour chaque email "i" card(Xi)=1000 ]
    #chaque email est représenté par un vecteur des fréquences d'apparition de chaque mot dans "mots" dans le texte de l'email
    listeX       = [  get_freq(x, mots) for x in emails ]

    #b. Liste des vecteurs Yi = list[ Yi=(y1, y2) pour chaque email "i" ]
    listeY       = [ [ abs(np.random.normal(0, 0.5)), abs(np.random.normal(0, 0.5))] for x in emails ]

    #c. On calcul sigma
    freqs        = [] #liste de toutes les frequences de tous les mots figurant dans tous les emails
    for xi in listeX:
        for y in xi:
            freqs.append(y)
    sigma        = np.std( np.array(freqs) )

    #d. Calcul des Pij et Qij
    listePij     = []
    for i in range(len(emails)):
        print ('Calcul des Pij pour le mail:', i)
        listePij.append([])
        for j in range(len(emails)):
            listePij[i].append( Pij(listeX, i, j, sigma) )

    #Etapes 3, 4, et 5
    epsilon       = 0.05 
    iteration     = 1
    max_iteration = len(listeY)*len(listeY)
    
    historique_C  = []
    C, listeQij   = calcul_c(listePij, listeY)
    print ('Initial C:', C)
    while 1:
        #a. etape 3
        yi = random.randint(0, len(listeY)-1)

        #b. etape 4
        print ('avant listeY[{}]:'.format(yi), listeY[yi])
        yi1, yi2      = gradient(listePij, listeQij, listeY, yi)
        

        listeY[yi][0] = listeY[yi][0]-epsilon*yi1
        listeY[yi][1] = listeY[yi][1]-epsilon*yi2
        
        print ('apres listeY[{}]:'.format(yi), listeY[yi])

        C, listeQij   = calcul_c(listePij, listeY)
        print ('Doing iteration {}\tC={}\n'.format(iteration, C))

        historique_C.append(C)

        #c. etape 5: convergence fonction définite plus haut
        if sne_converge(historique_C):
            break 

        iteration += 1
        if iteration > max_iteration:# une autre raison d'arreter le programme c'est lorsqu'on dépasse une valeur max d'iteration
            break
    
    print ('Finished {}'.format(iteration))
    plt.scatter([x[0] for x in listeY], [x[1] for x in listeY])# on affiche nos points
    plt.show()




"""#########################################Test SNE######################################################"""
def main():
    algo_sne()

if __name__ == '__main__':
    main() # appel à la fonction main qui teste l'algo sne
