#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#
# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
import random 
from numpy import inf

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...
    """
    def __init__(self):
       
        #self.purity=purityp
        #elf.exthreshold=exthreshold
        #self.maxdepth=maxdepth
       #Q self.tree=tree
        self.fidx=None
        self.fidx=None
        
        #print "   "        
        #pass
    def FindImpurity(self,Y):
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        if len(Count)!=1:
            Max=np.max(Count)
            Sum=np.sum(Count)
            Impurity=Max/Sum
            Label=UniqueLabels[np.where(Count==Max)]
        else:
            Impurity=1
            Label=UniqueLabels[0]

        return Label,Impurity
    

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...
        # YOUR CODE HERE
        best_point,score,Xlidx,Xridx=self.build_tree(X,Y,5)
        
            
        
        #---------End of Your Code-------------------------#
        return best_point,score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        if (X[self.fidx] < self.split):
            return True
        return False
    
    def build_tree(self, X, Y, depth):
        """ 
            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...
        """
        
        nexamples, nfeatures=X.shape
        # YOUR CODE HERE
        Split=0
        InfoGain=-float('Inf')
        RightChildInd=0
        LeftChildInd=0
        FeatureIndex=-1
        for i in range(nfeatures):
            Split_Temp,InfoGain_Temp,RightChildInd_Temp,LeftChildInd_Temp=self.evaluate_numerical_attribute(X[:,i],Y)
            if i!=0:
                if InfoGain_Temp>InfoGain:
                    Split=Split_Temp
                    InfoGain=InfoGain_Temp
                    RightChildInd=RightChildInd_Temp
                    LeftChildInd=LeftChildInd_Temp
                    FeatureIndex=i
            else:
                Split=Split_Temp
                InfoGain=InfoGain_Temp
                RightChildInd=RightChildInd_Temp
                LeftChildInd=LeftChildInd_Temp
                FeatureIndex=i
        #print("Node Created at Depth: ",depth,", Split: ",Split,", InfoGain: ",InfoGain,", FeatureIndex: ",FeatureIndex)
        return Split,InfoGain,LeftChildInd,RightChildInd
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        TotalEntropy=0
        TotalEntropy=self.TotalEntropy(Y)  
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        Index=0
        TargetedEntropy=float('Inf')
        
        for i in range(len(feat)):
            Point=feat[i] # index of the column 
            Temp=self.ComputeTargetEntropy(Point,feat,Y)
            if i!=0:
                if Temp<TargetedEntropy:
                    TargetedEntropy=Temp
                    Index=i
        score=TotalEntropy-TargetedEntropy
        split=feat[Index]
        RightChildInd=feat<split
        LeftChildInd=feat>=split
        return split, score, RightChildInd,LeftChildInd
            
        
        #---------End of Your Code-------------------------#
    def TotalEntropy(self,Y):
        # Done 
        UniqueLabels,Count=np.unique(Y,return_counts=True)
        TotalCount=np.sum(Count)
        TotalEntropy=0
        for i in range(len(UniqueLabels)):
            
            TotalEntropy+=(-Count[i]/TotalCount)*np.log2(Count[i]/TotalCount)
        return TotalEntropy
        
    def ComputeTargetEntropy(self,Point,feat,Y):
        
        TotalLabelsGreater,TotalLabelsLesser=Y[feat>=Point],Y[feat<Point]
        UniqueLabelsGreater,CountGreater=np.unique(TotalLabelsGreater,return_counts=True)
        UniqueLabelsLesser,CountLesser=np.unique(TotalLabelsLesser,return_counts=True)

        TotalGreaterCount,TotalLesserCount=np.sum(CountGreater),np.sum(CountLesser)
        TotalCount=TotalGreaterCount+TotalLesserCount
        result_1=0
        result_2=0
        for k in range(len(UniqueLabelsGreater)):
            
            
            if CountGreater[k]!=0 and TotalGreaterCount!=0:
                
                result_1+= ((-CountGreater[k]/TotalGreaterCount)*np.log2(CountGreater[k]/TotalGreaterCount))
        for k in range(len(UniqueLabelsLesser)):
            
            if CountLesser[k]!=0 and TotalLesserCount!=0:
                
                
                result_2+= ((-CountLesser[k]/TotalLesserCount)*np.log2(CountLesser[k]/TotalLesserCount))

        return result_1*(TotalGreaterCount/TotalCount) + result_2*(TotalLesserCount/TotalCount) 
    

class Random (WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        self.fidx=-1
        self.split=-1
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=np.round(np.sqrt(nfeatures))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        
        
        
            
        best_split=0
        best_score=0
        xlidx=None
        gain=0
        for i in range(self.nrandfeat):
            split,info_gain,left_temp,right_temp=np.findfindBestRandomSplit(X[:,np.random.choice(np.arange(0,self.nrandfeat))],Y)
        #---------End of Your Code-------------------------#
        
        
        
        
        return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        Entropy_klasses=weakLearner.ToalEntropy(Y)
        unique_splits=np.unique(feat)
        array1=[]
        for i in range(len(np.unique(feat))):
            if i!=0:
                array1.append((unique_splits[i]+unique_splits[i-1])/2)
        if len(array1)>self.nsplits:
            array1=np.random.choice(array1,self.nsplits,replace=False)
            
        return weakLearner.evaluate_numerical_attribute
        
        
            
            
        
        
        #--------Write Your Code Here ---------------------#
        
        
            
        
        #---------End of Your Code-------------------------#
    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy



# build a classifier ax+by+c=0
class LinearWeakLearner(Random):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr


    

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
            
        
        #---------End of Your Code-------------------------#
        
