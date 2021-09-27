#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:35:11 2020

@author: samaneh
## extracting various features from the sequences
##various window lengths are considered (7-35)
"""
from Bio import SeqIO
import pandas as pd
import numpy as np
import itertools

import os
if not os.path.exists('./Psites - data/'):
    os.makedirs('./Psites - data/')
    
prt = 'ACDEFGHIKLMPNTSVQRYW'
prta = ['A','C','D','E','F','G','H','I','K','L','M','P','N','T','S','V','Q','R','Y','W']

ploc = pd.read_csv('Final_Elham Khalili_40 % similarity_Data for  phosphorylation SOYBA  proteins.csv')
names = ploc[['Code id Protein ']].values
loc = ploc[['Positive Sites']].values
#names = names.astype(str)
seqNames= []
seqPrs= []

for record in SeqIO.parse("1613 sequences with 40%.txt", "fasta"):
    seqNames.append(record.id)
    seqPrs.append(record.seq)

for i in range(len(names)):
    ind = np.where(np.char.find(seqNames,names[i])>=0)[0]
    ind = ind[0]
    names[i] = seqNames[ind]
    
wlens = [7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]

for wlen in wlens:
    #############################################
    wlen2 = int((wlen-1)/2)
    posdf = pd.DataFrame(columns=('id','seq','loc'))
    negdf = pd.DataFrame(columns=('id','seq','loc'))
    for i in range(len(seqNames)):
        seq = seqPrs[i]
        ind = np.where(names == seqNames[i])[0] #finding postive locations
        ind = loc[ind]-1
        for j in range(len(seq)):
            if seq[j] == 'Y' or seq[j] == 'T' or seq[j] == 'S':
                if j-wlen2 >= 0: 
                    if j+wlen2+1 > len(seq):
                        wseq = seq[j-wlen2:]
                    else:
                        wseq = seq[j-wlen2:j+wlen2+1]
                else:
                    wseq = seq[:j+wlen2+1]
                if j in ind:
                    posdf = posdf.append({'id':seqNames[i], 'seq':wseq,'loc':j},ignore_index=True)
                else:
                    negdf = negdf.append({'id':seqNames[i], 'seq':wseq,'loc':j},ignore_index=True)
    
    
    ############################################## PSSM loading pos
    
    c1 = pd.read_csv('Class_1/Class_1.csv',header=None).values
    c2 = pd.read_csv('Class_2/Class_2.csv',header=None).values
    c3 = pd.read_csv('Class_3/Class_3.csv',header=None).values
    c4 = pd.read_csv('Class_4/Class_4.csv',header=None).values
    
    for i in range(len(c1)):
        ind = np.where(np.char.find(seqNames,c1[i,0])>=0)[0]
        ind = ind[0]
        c1[i,0] = seqNames[ind]
    
    for i in range(len(c2)):
        ind = np.where(np.char.find(seqNames,c2[i,0])>=0)[0]
        ind = ind[0]
        c2[i,0] = seqNames[ind]
        
    for i in range(len(c3)):
        ind = np.where(np.char.find(seqNames,c3[i,0])>=0)[0]
        ind = ind[0]
        c3[i,0] = seqNames[ind]
        
    for i in range(len(c4)):
        ind = np.where(np.char.find(seqNames,c4[i,0])>=0)[0]
        ind = ind[0]
        c4[i,0] = seqNames[ind]
        
    #for filename in os.listdir('/Class_1/pssm'):
    pssm_pos = np.zeros([len(posdf),20*wlen])
    for i in range(len(posdf)):
        ind = np.where(c1[:,0] == posdf['id'][i])[0]
        if len(ind) > 0: #file is in c1
            file = pd.read_csv('./Class_1/pssm/1ed99ee71935e3512b54182bd19f2935/1ed99ee71935e3512b54182bd19f2935_'+str(c1[ind,1][0])+'.pssm')
        else:
            ind = np.where(c2[:,0] == posdf['id'][i])[0]
            if len(ind) > 0: #file is in c1
                file = pd.read_csv('./Class_2/pssm/7b30c9b044389f1707d833d5e519392a/7b30c9b044389f1707d833d5e519392a_'+str(c2[ind,1][0])+'.pssm')
            else:
                ind = np.where(c3[:,0] == posdf['id'][i])[0]
                if len(ind) > 0: #file is in c1
                    file = pd.read_csv('./Class_3/pssm/0af60cc4a8044bac855a8a77d1904968/0af60cc4a8044bac855a8a77d1904968_'+str(c3[ind,1][0])+'.pssm')
                else:
                    ind = np.where(c4[:,0] == posdf['id'][i])[0]
                    if len(ind) > 0: #file is in c1
                        file = pd.read_csv('./Class_4/pssm/9c81bfa8dcf3cddaaa26c4e6d2686246/9c81bfa8dcf3cddaaa26c4e6d2686246_'+str(c4[ind,1][0])+'.pssm')
        p = np.zeros(20*wlen,)
        l = posdf['loc'][i] + 1      
        k=0
        for j in range(int(l-(wlen-1)/2),int(l+(wlen-1)/2+1)):
            if  0<j< len(file)-5:
                s = file['Last position-specific scoring matrix computed'][j].split(' ')
                s = np.array(s)
                s = s[np.where(s!='')[0]]
                s = s[2:22].astype(float)
            else:
                s = np.empty((20,))
                s[:] = np.nan
            p[k*20:(k+1)*20] = s
            k=k+1
        pssm_pos[i]= p
    
    
    pssm_pos = pd.DataFrame(data = pssm_pos, columns = prta *wlen)
    posdf_pssm = pd.concat([posdf,pssm_pos],ignore_index=True,axis=1)
    posdf_pssm.columns = ['id','seq','loc'] + prta *wlen
    
    ######################################### pssm neg
    pssm_neg = np.zeros([len(negdf),20*wlen])
    for i in range(len(negdf)):
        ind = np.where(c1[:,0] == negdf['id'][i])[0]
        if len(ind) > 0: #file is in c1
            file = pd.read_csv('./Class_1/pssm/1ed99ee71935e3512b54182bd19f2935/1ed99ee71935e3512b54182bd19f2935_'+str(c1[ind,1][0])+'.pssm')
        else:
            ind = np.where(c2[:,0] == negdf['id'][i])[0]
            if len(ind) > 0: #file is in c1
                file = pd.read_csv('./Class_2/pssm/7b30c9b044389f1707d833d5e519392a/7b30c9b044389f1707d833d5e519392a_'+str(c2[ind,1][0])+'.pssm')
            else:
                ind = np.where(c3[:,0] == negdf['id'][i])[0]
                if len(ind) > 0: #file is in c1
                    file = pd.read_csv('./Class_3/pssm/0af60cc4a8044bac855a8a77d1904968/0af60cc4a8044bac855a8a77d1904968_'+str(c3[ind,1][0])+'.pssm')
                else:
                    ind = np.where(c4[:,0] == negdf['id'][i])[0]
                    if len(ind) > 0: #file is in c1
                        file = pd.read_csv('./Class_4/pssm/9c81bfa8dcf3cddaaa26c4e6d2686246/9c81bfa8dcf3cddaaa26c4e6d2686246_'+str(c4[ind,1][0])+'.pssm')
        p = np.zeros(20*wlen,)
        l = negdf['loc'][i] + 1
        k=0
        for j in range(int(l-(wlen-1)/2),int(l+(wlen-1)/2+1)):
            if  0<j< len(file)-5:
                s = file['Last position-specific scoring matrix computed'][j].split(' ')
                s = np.array(s)
                s = s[np.where(s!='')[0]]
                s = s[2:22].astype(float)
            else:
                s = np.empty((20,))
                s[:] = np.nan
            p[k*20:(k+1)*20] = s
            k=k+1
        pssm_neg[i]= p
    
    pssm_neg = pd.DataFrame(data = pssm_neg, columns = prta*wlen)
    negdf_pssm = pd.concat([negdf,pssm_neg],ignore_index=True,axis=1)
    negdf_pssm.columns = ['id','seq','loc'] + prta *wlen
        
    ################################### pos samples
    
    alltwo = [p for p in itertools.product(prt, repeat=2)]
    alltwo =  np.array([''.join(i) for i in alltwo])
    allthree =[p for p in itertools.product(prt, repeat=3)]
    allthree = np.array([''.join(i) for i in allthree] )
    
    seqs = posdf['seq'].values
    onemer = np.zeros([len(seqs),20])
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)):
            ind = prt.find(sq[j])
            onemer[i,ind] = onemer[i,ind] + 1
        onemer[i,:] = onemer[i,:] / len(sq)
    
    twomer = np.zeros([len(seqs),400])        
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)-1):
            ind = np.where(alltwo == sq[j]+sq[j+1])[0][0]
            twomer[i,ind] = twomer[i,ind] + 1
        twomer[i,:] = twomer[i,:] / len(sq)
            
    threemer = np.zeros([len(seqs),8000])        
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)-2):
            ind = np.where(allthree == sq[j]+sq[j+1]+sq[j+2])[0][0]
            threemer[i,ind] = threemer[i,ind] + 1
        threemer[i,:] = threemer[i,:] / len(sq)
            
    ponemer = pd.DataFrame(data = onemer, columns = prta)
    ptwomer = pd.DataFrame(data = twomer, columns = alltwo)
    pthreemer = pd.DataFrame(data = threemer, columns = allthree)
    
    
    posdf_onemer = pd.concat([posdf,ponemer],ignore_index=True,axis=1)
    posdf_onemer.columns = ['id','seq','loc'] + prta
    posdf_twomer = pd.concat([posdf,ptwomer],ignore_index=True,axis=1)
    posdf_twomer.columns = ['id','seq','loc'] + list(alltwo)
    posdf_threemer = pd.concat([posdf,pthreemer],ignore_index=True,axis=1)
    posdf_threemer.columns = ['id','seq','loc'] + list(allthree)
    ############################# neg samples
    seqs = negdf['seq'].values
    onemer = np.zeros([len(seqs),20])
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)):
            ind = prt.find(sq[j])
            onemer[i,ind] = onemer[i,ind] + 1
        onemer[i,:] = onemer[i,:] / len(sq)
    
    twomer = np.zeros([len(seqs),400])        
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)-1):
            ind = np.where(alltwo == sq[j]+sq[j+1])[0][0]
            twomer[i,ind] = twomer[i,ind] + 1
        twomer[i,:] = twomer[i,:] / len(sq)
            
    threemer = np.zeros([len(seqs),8000])        
    for i in range(len(seqs)):
        sq = seqs[i]
        for j in range(len(sq)-2):
            ind = np.where(allthree == sq[j]+sq[j+1]+sq[j+2])[0][0]
            threemer[i,ind] = threemer[i,ind] + 1
        threemer[i,:] = threemer[i,:] / len(sq)
            
    ponemer = pd.DataFrame(data = onemer, columns = prta)
    ptwomer = pd.DataFrame(data = twomer, columns = alltwo)
    pthreemer = pd.DataFrame(data = threemer, columns = allthree)
    
    negdf_onemer = pd.concat([negdf,ponemer],ignore_index=True,axis=1)
    negdf_onemer.columns = ['id','seq','loc'] + prta
    negdf_twomer = pd.concat([negdf,ptwomer],ignore_index=True,axis=1)
    negdf_twomer.columns = ['id','seq','loc'] + list(alltwo)
    negdf_threemer = pd.concat([negdf,pthreemer],ignore_index=True,axis=1)
    negdf_threemer.columns = ['id','seq','loc'] + list(allthree)
    
    print(len(negdf_onemer),len(negdf_twomer),len(negdf_threemer),len(negdf_pssm))
    print(len(posdf_onemer),len(posdf_twomer),len(posdf_threemer),len(posdf_pssm))
    posdf_onemer.to_csv('./Psites - data/pos_onemer_'+str(wlen)+'.csv')
    negdf_onemer.to_csv('./Psites - data/neg_onemer_'+str(wlen)+'.csv')
    
    posdf_twomer.to_csv('./Psites - data/pos_twomer_'+str(wlen)+'.csv')
    negdf_twomer.to_csv('./Psites - data/neg_twomer_'+str(wlen)+'.csv')
    
    posdf_threemer.to_csv('./Psites - data/pos_threemer_'+str(wlen)+'.csv')
    negdf_threemer.to_csv('./Psites - data/neg_threemer_'+str(wlen)+'.csv')
    
    posdf_pssm.to_csv('./Psites - data/pos_pssm_'+str(wlen)+'.csv')
    negdf_pssm.to_csv('./Psites - data/neg_pssm_'+str(wlen)+'.csv')
                
    del posdf,posdf_onemer, posdf_pssm, posdf_threemer, posdf_twomer
    del negdf, negdf_onemer, negdf_pssm, negdf_threemer, negdf_twomer
    del ponemer, pthreemer, ptwomer