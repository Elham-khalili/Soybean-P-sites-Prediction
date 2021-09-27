import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt
import warnings
from tabnettrain import *
from common_functions import *
#from sdae_test import *
warnings.simplefilter("ignore")

##calculated performance for various classifiers and wndow sizes

def testv1(method ='rf',wlen=21, feat_type='1mer'):
    
    accm = []
    senm = []
    spsm = []
    roc_aucm = []
    npvm = []
    prsm = []
    f1scorem = []
    tprs = []
    mccm = []
    mean_fpr = np.linspace(0, 1, 100)
    
    #load data
    pfeat, nfeat = feat_prep(feat_type,wlen)
    nfeat = nfeat.dropna().reset_index(drop=True)
    pfeat = pfeat.dropna().reset_index(drop=True)
         
    #features names
    feat = list(pfeat)
   # print(len(feat))
  #  feat = feat[4:]
   
    
    #convert it to np array
    pfeat = pfeat.values
    nfeat = nfeat.values

    
    #features and labels
    pfeatl = np.zeros((len(pfeat),))
   # pfeat = pfeat[:,4:]
    nfeatl = np.ones((len(nfeat),))
  #  nfeat = nfeat[:,4:]
    
    num_itr = 1
    wdata_all = np.concatenate((pfeat,nfeat))
    wlabel_all = np.concatenate((pfeatl,nfeatl))
        
    est_labels=np.zeros([num_itr,len(pfeatl)*2])
    
    #cross validation and apply multiple machine learning techniques
    for i in range(num_itr):
        cv=StratifiedKFold(n_splits=10, random_state=None, shuffle=False) # 5fold cross validation
        
        wdata, wlabel = balanced_subsample(wdata_all,wlabel_all)
        
        stratified_5folds = cv.split(wdata, wlabel) #five folds
        importances =[]
        
        for trind, teind in stratified_5folds:
            #80% of the data            
            tr=wdata[trind] 
            trl=wlabel[trind]
            # 20% of the data for final test
            te=wdata[teind]
            tel=wlabel[teind]
            
            
            if method == 'rf':
               model = RandomForestClassifier(n_estimators= 60, max_depth=2, random_state=0).fit(tr,trl)
               importances.append(model.feature_importances_)
            if method == 'gbt':
               model = GradientBoostingClassifier(n_estimators= 60, max_depth=2, random_state=0).fit(tr,trl)
               importances.append(model.feature_importances_)
            elif method == 'lr-l1':
               model = LogisticRegression(penalty='l1',solver='liblinear').fit(tr,trl)
               importances.append(np.reshape(model.coef_,(len(feat,))))
            elif method == 'lr-l2':
               model = LogisticRegression(penalty='l2').fit(tr,trl)
               importances.append(np.reshape(model.coef_,(len(feat,))))
            elif method == 'svm':
               model = svm.SVC(kernel = 'linear',probability=True).fit(tr,trl)
               importances.append(np.reshape(model.coef_,(len(feat,))))
            elif method == 'tabnet':
                model = TabNetTest(tr,trl)
                importances.append(model.feature_importances_)

            pred = model.predict_proba(te)[:,1]
            
            thr = find_thr(model.predict_proba(tr)[:,1],trl)
            prr = np.where(pred > thr, 1, 0)
            est_labels[i,teind] = prr
            
            acc,sen,sps,roc_auc,prs,npv,f1score,mcc = performance_calculation(tel,prr,pred)
            accm.append(100*acc)
            senm.append(100*sen)
            spsm.append(100*sps)
            roc_aucm.append(100*roc_auc)
            prsm.append(100*prs)
            npvm.append(100*npv)
            f1scorem.append(100*f1score)
            mccm.append(100*mcc)
            fpr, tpr, thresholds = roc_curve(tel, pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
    
    imp = np.mean(importances,axis = 0)
    indices = np.argsort(np.mean(importances,axis = 0))[::-1]
    sorted_features = []
    for f in range(len(indices)):
        sorted_features.append(feat[indices[f]])
    #save confusion_matrix
    cm = np.zeros([2,2])
    for i in range(num_itr):
        cm = cm + confusion_matrix(wlabel,est_labels[i,:])
    cm=np.floor(cm/num_itr)
  #  plot_confusion_matrix(cm,['Neg','Pos'],'./results/' +method +'_'+str(wlen)+'_'+feat_type+ '_confusion_matrix.eps')
    
    #save roc curve
#    plt.figure()
#    mean_tpr = np.mean(tprs, axis=0)
#    mean_tpr[-1] = 1.0
 #   mean_auc = auc(mean_fpr, mean_tpr)
 #   std_auc = np.std(roc_aucm)
 #   np.savez('./results/' +method +'_'+str(wlen)+'_'+feat_type+ '_roc',tpr = mean_tpr, fpr = mean_fpr)
#    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
#    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#    std_tpr = np.std(tprs, axis=0)
#    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
  #  plt.xlabel('False Positive Rate (1 - Specificity)')
  #  plt.ylabel('True Positive Rate (Sensitivity)')
    
 #   plt.savefig('./results/' +method +'_'+str(wlen)+'_'+feat_type+ '_roccurve.eps', format='eps', dpi=1000)
#    plt.close()
    del pfeat,nfeat,wdata
    return imp,sorted_features, accm, senm*100, spsm*100, roc_aucm*100, prsm*100, npvm*100, f1scorem*100, est_labels,mccm
    #print(method + ': ', np.mean(accm),np.std(accm),np.mean(senm),np.std(senm),np.mean(spsm),np.std(spsm),np.mean(roc_aucm),np.std(roc_aucm))

def feat_prep(feat_type='All',wlen=21):
    
    pdf_1mer = pd.read_csv('./Psites - data/pos_onemer_'+str(wlen)+'.csv')
    pdf_2mer = pd.read_csv('./Psites - data/pos_twomer_'+str(wlen)+'.csv')
    pdf_3mer = pd.read_csv('./Psites - data/pos_threemer_'+str(wlen)+'.csv')
    pdf_pssm = pd.read_csv('./Psites - data/pos_pssm_'+str(wlen)+'.csv')


    ndf_1mer = pd.read_csv('./Psites - data/neg_onemer_'+str(wlen)+'.csv')
    ndf_2mer = pd.read_csv('./Psites - data/neg_twomer_'+str(wlen)+'.csv')
    ndf_3mer = pd.read_csv('./Psites - data/neg_threemer_'+str(wlen)+'.csv')
    ndf_pssm = pd.read_csv('./Psites - data/neg_pssm_'+str(wlen)+'.csv')
    
    pdf_1mer = pdf_1mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    ndf_1mer = ndf_1mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    pdf_2mer = pdf_2mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    ndf_2mer = ndf_2mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    pdf_3mer = pdf_3mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    ndf_3mer = ndf_3mer.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    pdf_pssm = pdf_pssm.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    ndf_pssm = ndf_pssm.drop(columns = ['Unnamed: 0','loc','id','seq'],axis=1)
    
    if feat_type == '1mer':
        pfeat = pdf_1mer
        nfeat = ndf_1mer
    elif feat_type == '2mer':
        pfeat = pdf_2mer
        nfeat = ndf_2mer
    elif feat_type == '3mer':
        pfeat = pdf_3mer
        nfeat = ndf_3mer
    elif feat_type == '1-2mer':
        pfeat = pd.concat([pdf_1mer,pdf_2mer],ignore_index=True,axis=1)
        nfeat = pd.concat([ndf_1mer,ndf_2mer],ignore_index=True,axis=1)
    elif feat_type == '1-2-3mer':
        
        pfeat = pd.concat([pdf_1mer,pdf_2mer,pdf_3mer],ignore_index=True,axis=1)
        nfeat = pd.concat([ndf_1mer,ndf_2mer,ndf_3mer],ignore_index=True,axis=1)
    elif feat_type == 'pssm':
        pfeat = pdf_pssm
        nfeat = ndf_pssm
    elif feat_type == 'All':
        pfeat = pd.concat([pdf_1mer,pdf_2mer,pdf_3mer,pdf_pssm],ignore_index=True,axis=1)
        nfeat = pd.concat([ndf_1mer,ndf_2mer,ndf_3mer,ndf_pssm],ignore_index=True,axis=1)
    del pdf_1mer,pdf_2mer,pdf_3mer,pdf_pssm, ndf_1mer,ndf_2mer,ndf_3mer,ndf_pssm
    pfeat = pfeat.astype(np.float16)
    nfeat = nfeat.astype(np.float16)
    return pfeat, nfeat

methods = ['tabnet']#,'lr-l1','lr-l2','rf','gbt']
features_type =['All']#,'1mer','2mer','3mer','1-2mer','1-2-3mer','pssm']
wlen = [13]#7,9,11,13,17,19,21,23,25,27,29,31,33,35]
for d in wlen:
    file = open('./' +str(d) + '_performance.txt',"a")
    file.write('Method, Accuracy, Sensitivity, Specificity, AUC, Precision, Negative Predictive Value, F1 Score, MCC \n')
    for m in methods:
        for f in features_type:
            print(d,m,f)
            imp,sorted_features, accm, senm, spsm, roc_aucm, prsm, npvm, f1scorem, est_labels,mccm = testv1(m,d,f)
            print(m,f, np.mean(roc_aucm))
            file.write('method: '+ m + 'features_type_' + f + ':, ')
            file.write("%0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f, %0.2f \u00B1 %0.2f,  %0.2f \u00B1 %0.2f \n" % (np.mean(accm),np.std(accm),np.mean(senm),np.std(senm),np.mean(spsm),np.std(spsm),np.mean(roc_aucm),np.std(roc_aucm),np.mean(prsm),np.std(prsm),np.mean(npvm),np.std(npvm),np.mean(f1scorem),np.std(f1scorem),np.mean(mccm),np.std(mccm)))
            df = pd.DataFrame(data={"features": sorted_features})
            df.to_csv('./' +m +'_'+str(d)+'_'+f+'_ranked-features.csv', sep=',',index=False)
            np.savetxt('./' +m+'_'+str(d)+'_'+f +'_estimatedlabels.csv', est_labels.astype(np.int),delimiter=', ',fmt='%d')
            np.savetxt('./' +m+'_'+str(d)+'_'+f +'imp.csv', imp,delimiter=', ',fmt='%f')
    file.close()