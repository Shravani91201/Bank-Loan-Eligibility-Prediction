#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: 
# 
# A Company wants to automate the loan eligibility process based on customer details provided while filling online application form. The details filled by 
# the customer are Gender, Marital Status, Education, Number of Dependents, Income of self and co applicant, Required Loan Amount, Required Loan Term, Credit History and others.
# The requirements are as follows:
# 
# 1.)Check eligibility of the Customer given the inputs described above.(Classification)

# # read dataset

# In[2]:


import pandas as pd 
from warnings import filterwarnings
filterwarnings("ignore")
A=pd.read_csv("training_set.csv")


# In[3]:


A.head(10)


# In[4]:


A.info()


# In[5]:


A.describe()


# In[6]:


A=A.replace(to_replace='3+',value=4)


# In[7]:


A.Dependents


# In[8]:


A['Dependents']=A['Dependents'].astype(float)
print(A.dtypes)


# # Define Fuctions

# In[9]:


def catconsep(df):
    cat = []
    con = []
    for i in df.columns:
        if(df[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)
    return cat,con


# In[10]:


def replacer(df):
    cat,con = catconsep(df)
    for i in con:
        x = df[i].mean()
        df[i]=df[i].fillna(x)

    for i in cat:
        x = df[i].mode()[0]
        df[i]=df[i].fillna(x)


# In[11]:


def standardize(df):
    import pandas as pd
    cat,con = catconsep(df)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X1 = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    return X1


# In[12]:


def outliers(df):
    df = standardize(df)
    outliers = []
    cat,con = catconsep(df)
    for i in con:
        outliers.extend(list(df[df[i]>3].index))
        outliers.extend(list(df[df[i]<-3].index))
    from numpy import unique
    Q = list(unique(outliers))
    return Q


# In[13]:


def preprocessing(df):
    cat,con = catconsep(df)
    from sklearn.preprocessing import MinMaxScaler
    ss = MinMaxScaler()
    import pandas as pd
    X1 = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    X2 = pd.get_dummies(df[cat])
    Xnew = X1.join(X2)
    return Xnew


# In[81]:

def ANOVA(df,cat,con):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    eqn = str(con) + " ~ " + str(cat)
    model = ols(eqn,df).fit()
    from statsmodels.stats.anova import anova_lm
    Q = anova_lm(model)
    return round(Q.iloc[0:1,4:5].values[0][0],5)



# In[82]:

Q = ["Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"]
for i in Q:
    print(i,ANOVA(A,"Loan_Status",i))



# In[15]:


def chisq(df,cat1,cat2):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df[cat1],df[cat2])
    a,b,c,d = chi2_contingency(ct)
    return round(b,5)


# # missing data tretment

# In[16]:


Q=A.isna().sum()
Q[Q>0]


# In[17]:


replacer(A)


# In[18]:


Q=A.isna().sum()
Q[Q>0]


# # outliers

# In[19]:


A.shape


# In[20]:


out=outliers(A)
len(out)


# In[21]:


A=A.drop(index=out,axis=0)


# In[22]:


A.index=range(0,A.shape[0],1)


# In[23]:


A.shape


# # 1.)Check eligibility of the Customer given the inputs described above.(Classification)

# In[24]:


A.nunique()


# # X and Y define

# In[25]:


Y=A[["Loan_Status"]]
X=A.drop(labels=["Loan_Status","Loan_ID"],axis=1)


# # cat,con seprater

# In[26]:


cat,con=catconsep(X)
cat,con


# In[27]:


con.remove("Credit_History")
cat.append("Credit_History")


# In[28]:


con


# In[29]:


cat


# # Visulization

# # ANOVA

# In[79]:




# # Chisquare test

# In[32]:


Q = ["Loan_ID",
      "Gender",
      "Married",
      "Education",
      "Self_Employed",
       "Property_Area",
       "Credit_History"]
for i in Q:
    print(i,chisq(A,"Loan_Status",i))


# # preprocessing

# In[33]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
X2 = pd.get_dummies(X[cat])
Xnew = X1.join(X2)
Xnew.head(3)


# # split

# In[34]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=31)


# # crete model function

# In[35]:


def modeller(mo):
    model = mo.fit(xtrain,ytrain)
    pred_tr = model.predict(xtrain)
    pred_ts = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = round(accuracy_score(ytrain,pred_tr),2)
    ts_acc = round(accuracy_score(ytest,pred_ts),2)
    return tr_acc,ts_acc


# # KNN model

# In[36]:


import numpy as np
xtrain = np.ascontiguousarray(xtrain)
xtest = np.ascontiguousarray(xtest)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
modeller(knn)


# In[38]:



tr = []
ts = []
for i in range(2,40,1):
    knn = KNeighborsClassifier(n_neighbors=i)
    t1,t2=modeller(knn)
    tr.append(t1)
    ts.append(t2)
    
    
import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# In[39]:


knn = KNeighborsClassifier(n_neighbors=5)
modeller(knn)


# # logistic regression

# In[40]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
modeller(lr)


# # Decision tree

# In[41]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=31)
modeller(dtc)


# In[42]:


tr = []
ts = []
for i in range(2,40,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(max_depth=i)
    t1,t2=modeller(dtc)
    tr.append(t1)
    ts.append(t2)
    
    
import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3)
modeller(dtc)


# # Random Forest Classifier

# In[44]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=31)
modeller(rfc)


# In[45]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15,max_depth=2)
modeller(rfc)


# # AdaBoost Classifier

# In[46]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(DecisionTreeClassifier(random_state=31,max_depth=2),n_estimators=10)
modeller(abc)


# In[47]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(DecisionTreeClassifier(random_state=31,max_depth=2),n_estimators=2)
modeller(abc)


# # Model Result

# KNN= 0.81, 0.79
# 
# logistic regression= 0.81, 0.84
# 
# Decision tree= 0.82, 0.84
# 
# Random Forest Classifier= 0.78, 0.78
# 
# AdaBoost Classifier= 0.81, 0.84

# # Train the model with entire Data

# In[48]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3)
model=dtc.fit(Xnew,Y)


# # test prediction

# In[49]:


B=pd.read_csv("testing_set.csv")


# In[50]:


B.head(2)  


# In[51]:


B.info()


# In[52]:


B=B.replace(to_replace='3+',value=4)


# In[53]:


B['Dependents']=B['Dependents'].astype(float)
print(B.dtypes)


# # NA values in test set

# In[54]:


for i in B.columns:
    if(B[i].dtype=="object"):  ##B column
        X=A[i].mode()[0]   ##A colm mode
        B[i]=B[i].fillna(X) ##fill in B
    else:
        X=A[i].mean()  ## A mean fill in B
        B[i]=B[i].fillna(X)


# In[55]:


Q=B.isna().sum()
Q[Q>0]


# In[56]:


Q=B.drop(labels=["Loan_ID","Gender"],axis=1)
Q.head(1)


# In[57]:


cat,con=catconsep(Q)
cat,con


# In[58]:


con.remove("Credit_History")
cat.append("Credit_History")


# In[59]:


cat


# # preprocessing of test set

# In[60]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X11=pd.DataFrame(ss.fit_transform(Q[con]),columns=con)
X21=pd.get_dummies(Q[cat])
Xnew2=X11.join(X21)


# In[61]:


Xnew2.head(2)


# # Prediction

# In[62]:


#Dependents', 'Gender_Female', 'Gender_Male


# In[63]:


len(Xnew.columns),len(Xnew2.columns)


# In[64]:


Xnew2['Dependents']=0
Xnew2['Gender_Female']=0
Xnew2['Gender_Male']=0


# In[65]:


final_DF = Xnew2[Xnew.columns]
final_DF.head(1)


# In[66]:


pred = model.predict(final_DF)
final=pd.DataFrame(pred)
final.columns=['Loan_Status']
final['Loan_ID']=B.Loan_ID


# In[67]:


final=final[["Loan_ID","Loan_Status"]]
final


# # New DF with Result

# In[68]:


B1=B
B1['Loan_Status']=pred
B1.head(10)


# In[69]:


B1.to_csv("loan_file.csv")


# In[70]:


import pickle as p


# In[71]:


p.dump(model,open("Model1.pkl","wb"))


# In[72]:


pickle_model = p.load(open("Model1.pkl", 'rb'))


# #prediction trial on randam values 

# In[73]:


pickle_model.predict([[-0.693383,0.398479,-0.860843,0.239408,1.0,0,1,1,0,1,0,1,0,0,0,1,0]])


# In[76]:




# In[ ]:





# In[ ]:





# In[ ]:




