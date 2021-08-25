from scipy.stats import binned_statistic,binned_statistic_2d
from tqdm import tqdm
from sklearn.manifold import TSNE
from collections import defaultdict
from scipy.stats import pearsonr,spearmanr

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import ast
import networkx as nx


class Trajectory_Analysis:
    
    
    def __int__(self,traj_df):
        
        self.trajDF = traj_df
        
        self.days = traj_df['Days'] = (traj_df['created'] - traj_df.iloc[0]['created']).dt.days
        self.trajectory = traj_df[['X_tSNE','Y_tSNE']].values
        
        
    

    def jump_distances_vector(self,trajectory):
        
        j_dist = np.sqrt(np.sum((self.trajectory[1:]-self.trajectory[:-1])**2,axis=1))

        return j_dist

    def z_scoreD(self,dist_vec):
        
        if type(dist_vec) != np.ndarray:
            dist_vec = np.array(dist_vec)

        u,sig = np.mean(dist_vec),np.std(dist_vec)
        zscore = (dist_vec-u)/sigma

        return zscore

    def distances_fromOrigin_vector(self,trajectory):
        
        dist0 = np.sqrt(np.sum((b-a[0])**2,axis=1))

        return dist0    

    def jump_duration_days(self,days):
        
        if type(days) != np.ndarray:
            days = np.array(days)
        
        timeINdays=days[1:]-days[:-1]
        
        return timeINdays

    def mass_pos_loc(self):
        '''
        This will return the mass dictionary of the locations (Mass here is the number of visits on a location 
        by an author) of all the visited positions in the embedded space and the position of 
        the origin of the trajectoy for an author. 
        '''

        mass_dict = defaultdict(int)
        Pos = {}
        for i in range(len(self.trajDF)):

            tags = self.trajDF.iloc[i]['categories'].split(' ')
            x,y = self.trajDF.iloc[i]['X_tSNE'],self.trajDF.iloc[i]['Y_tSNE'] 
    #         x,y = self.trajDF.iloc[i]['X_cm'],self.trajDF.iloc[i]['Y_cm'] 

            tags.sort()  #### we sort and create categories in alphabetical order so that 'x y' is same as 'y x'
                         #### since they are the same location in embedded space
            
            l = ''
            for j in tags:
                l+='%s '%(j)

            l = l.strip()

            if l not in Pos:
                Pos[l] = (x,y)

            mass_dict[l]+=1

        origin = (self.trajDF.iloc[0]['X_tSNE'],self.trajDF.iloc[0]['Y_tSNE'])
    #     origin = (self.trajDF.iloc[0]['X_cm'],self.trajDF.iloc[0]['Y_cm'])

        mass_dict = {k: v for k, v in sorted(mass_dict.items(), key=lambda item: item[1],\
                                             reverse=True)}

        return mass_dict,Pos,origin


    def radius_of_gyration(self):

        '''
        Radius of gyration for an authors' trajectory
        '''

        DF = self.trajDF

        loc = []

        mass_dict,Pos,Org = mass_pos_loc(DF)
        x0,y0=Org
        M = sum(mass_dict.values())
        xcm,ycm,rg = 0,0,0
        for k in mass_dict:

            mi = mass_dict[k]
            xi,yi = Pos[k]

            xcm += (xi-x0)*mi
            ycm += (yi-y0)*mi

        rcm = (xcm/M,ycm/M)

        for k in mass_dict:

            mi = mass_dict[k]
            xi,yi = Pos[k]

            rc = np.sqrt(rcm[0]**2 + rcm[1]**2)
            ri = np.sqrt((xi-x0)**2 + (yi-y0)**2)

            rg += mi*((ri-rc)**2)

        RG = np.sqrt(rg/M)
        return rcm, RG, Org    
    

    def radius_of_gyration_k(self,k):

        '''
        Radius of gyration about the k-th most visited location
        '''

        DF = self.trajDF

        k = k

        loc = []

        mass_dict,Pos,Org = mass_pos_loc(DF)

        x0,y0=Org

        top_loc = list(mass_dict.keys())

        top_kth_massDict = {}

        for kloc in top_loc[:k]:
            top_kth_massDict[kloc]=mass_dict[kloc]

        M = sum(top_kth_massDict.values())

        xcm,ycm,rg = 0,0,0
        for i in top_kth_massDict:

            mi = top_kth_massDict[i]
            xi,yi = Pos[i]

            xcm += (xi-x0)*mi
            ycm += (yi-y0)*mi

        kth_rcm = (xcm/M,ycm/M)

        for j in top_loc[:k]:

            mi = mass_dict[j]
            xi,yi = Pos[j]

            rc = np.sqrt(kth_rcm[0]**2 + kth_rcm[1]**2)
            ri = np.sqrt((xi-x0)**2 + (yi-y0)**2)

            rg += mi*((ri-rc)**2)

        RGk = np.sqrt(rg/M)
        return kth_rcm, RGk    