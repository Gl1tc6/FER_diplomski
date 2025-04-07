import sklearn.svm as svm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data as d
import matplotlib.pyplot as plt

class KSVMWrap:
    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''
    
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svm = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)
    
    def predict(self, X):
        return self.svm.predict(X)
    
    def get_scores(self, X):
        return self.svm.decision_function(X)
    
    def support(self):
        return self.svm.support_
    
if __name__ == "__main__":
    np.random.seed(100)
    
    # Get data
    X, Y_ = d.sample_gmm_2d(6, 2, 10)
    model = KSVMWrap(X=X, Y_=Y_, param_svm_c=1, param_svm_gamma='auto')
    
    # Evaluate
    preds = model.predict(X)
    acc, prec_rec, _ = d.eval_perf_multi(preds, Y_)
    
    print(f'Accuracy: {acc}\n')
    for c, (prec, rec) in enumerate(prec_rec):
        print(f"Class {c}\nPrecision:\t{prec}\nRecall:\t{rec}\n")
    
    # Plot decision boundary
    decfun = model.get_scores
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    d.graph_surface(decfun, rect, offset=0)
    d.graph_data(X, Y_, preds, special=model.support())
    plt.show()