# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:13:28 2022

@author: owner
"""

# Importation of necessary packages
from scipy import stats
import random
import numpy as np


class Params:
    def __init__(self,m_stage,n_jobs,k_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm):
        self.m_stage=m_stage
        self.n_jobs=n_jobs
        self.k_sen=k_sen
        self.mu_Ar=mu_Ar
        self.mu_Ai=mu_Ai
        self.mu_Dint=mu_Dint
        self.Mu_mu_Xm=Mu_mu_Xm
        self.Mu_sigma_Xm=Mu_sigma_Xm
        self.Mu_mu_Zm=Mu_mu_Zm
        self.Mu_sigma_Zm=Mu_sigma_Zm

class Inst:
    def __init__(self,A,D,X,Z):
        self.A=A
        self.D=D
        self.X=X
        self.Z=Z
class Inpt:
    def __init__(self,muA,stdA,muDint,stdDint,muX,stdX,muZ,stdZ):
        self.muA=muA
        self.stdA=stdA
        self.muDint=muDint
        self.stdDint=stdDint
        self.muX=muX
        self.stdX=stdX
        self.muZ=muZ
        self.stdZ=stdZ
class Sched:
    def __init__(self,job,Y,S,Idle):
        self.job=job
        self.Y=Y
        self.S=S
        self.Idle=Idle
