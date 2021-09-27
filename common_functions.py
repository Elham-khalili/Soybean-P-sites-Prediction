import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def balanced_subsample(x,y):
    class_xs = []
    min_elems = None
    subsample_size=1.0 # keep equal number of samples for all classes
    
    for yi in np.unique(y): # for any class in our data
        elems = x[(y == yi)] #find the data of each label
        class_xs.append((yi, elems)) 
        if min_elems == None or elems.shape[0] < min_elems: # find min_elemns as the group with lower samples
            min_elems = elems.shape[0]

    use_elems = min_elems # if proportion is not 1 not applicable in our work
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs: # randomly select samples from larger group
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        
        xs.append(x_)
        ys.append(y_)
        
        
    # concatenate the data of all groups before returning them 
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    
    return xs,ys

    
def tsne_plot(numComp):
    hd = pd.read_csv('Data-H2.csv')
    nhd = pd.read_csv('Data-NH2.csv')
    hd = hd.dropna()
    nhd = nhd.dropna()
    
    #convert it to np array
    hd = hd.values
    nhd = nhd.values
    hdf = hd[:,1:]
    nhdf = nhd[:,1:]
    data = np.concatenate((hdf,nhdf))
    
    #features and labels
    hdl = np.zeros((len(hd),))
    nhdl = np.ones((len(nhd),))
    labels = np.concatenate((hdl,nhdl))
    
    sne = manifold.TSNE(numComp, random_state=0)
    rdata = sne.fit_transform(data)
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    
    target_ids = range(2)
    tl = labels
    colors = ['darkgreen','orange']
    for i, c1, label in zip(target_ids, colors, ['Healthy', 'Infected']):
        plt.scatter(rdata[tl == i, 0], rdata[tl == i, 1], c=c1, label=label)
            
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
    lgd= ax.legend(loc='upper center', bbox_to_anchor=(1.2, 1), shadow=True, ncol=1)
    plt.savefig('tsne_plot.eps', format='eps', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=1000)
    plt.close(fig)
#calculate the performance (Accuracy, sensitivity, specificity, AUC)

def performance_calculation(array1,array2,array3):
     tn, fp, fn, tp = confusion_matrix(array1,array2).ravel()
     total=tn+fp+fn+tp
     acc= (tn+tp)/total
     sen = tp/(tp+fn)
     sps = tn/(tn+fp)
     prs = tp/(tp+fp)
     npv = tn/(tn+fn)
     f1score = 2* (sen*prs)/(prs+sen)
     fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
     roc_auc=metrics.auc(fpr, tpr)
     mcc =  matthews_corrcoef(array1,array2)
     
     return acc,sen,sps,roc_auc,prs,npv,f1score,mcc

def find_thr(pred,label):
    
    #find the best threshold where false possitive rate and falsi negative points cross
    minn=100000
    thrr=0.4
    
    for thr in np.arange(0.1,1,0.05):
        prr = np.where(pred > thr, 1, 0)
        tn, fp, fn, tp = confusion_matrix(label,prr).ravel()
        if tp+fn > 0:
            frr=fn/(tp+fn)
        else:
            frr = 0
        if tn+fp > 0:    
            far=fp/(tn+fp)
        else:
            far = 0 
        if np.abs(frr - far) < minn:
            minn=np.abs(frr - far)
            thrr=thr
            
    return thrr
 
def plot_confusion_matrix(cm, classes, name, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(name, format='eps', dpi=1000)
    plt.close()
