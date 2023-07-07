#!/usr/bin/env python
# coding: utf-8

# ### **Credit Risk Predict with Machine Learning**

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')
     


# ### **Import Dataset**

# In[2]:


# import data 
df = pd.read_csv('loan_data_2007_2014.csv')
# sample
df.sample(5)


# ### **Data Exploration**

# In[3]:


# shiw raw of data and feature numbers
print("Total rows :", df.shape[0])
print("Total features: ", df.shape[1])


# In[4]:


# chek duplicated value
df.duplicated().sum()


# In[5]:


#show data item
list_item= []
for col in df.columns :
    list_item.append([col, df[col].dtype, df[col].isna().sum(), 100*df[col].isna().sum()/len(df[col]), df[col].nunique(), df[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_item, columns='Feature,Data Type,  Null, Null(%), Unique, Unique Sample'.split(","))
desc_df


# #### There are some features which considered to be dropped
# 
# - *Unnamed: 0, id, and member_id* are representing unique values for each rows.
# - policy code, and application type have only 1 value.
# - unnecessary features like title, url, zip_code, addr_state, and desc.
# - sub_grade seems like a form of expert judgement from grade
# - 100% null values like annual_inc_joint, dti_joint, verification_status_joint, open_acc_6m, open_il_6m, open_il_12m, - -- open_il_24m, mths_since_rcnt_il, total_bal_il, il_util, open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi,total_cu_tl, and inq_last_12m.
# 
# #### Some features with > 20% of missing values will have a higher probability of being dropped since performing imputation seems to have a huge impact on originality of the dataset.
# 
# - mths_since_last_delinq, mths_since_last_record, next_pymnt_d,mths_since_last_major_derog
# 
# #### Some features with < 20% of missing values will be maintained for imputation if possible
# 
# - emp_title, emp_length, acc_now_delinq
# 
# #### Some features with date-like values will be converted to datetime
# 
# - last_pymnt_d, next_pymnt_d, last_credit_pull_d, earliest_cr_line
# 
# #### Some feature's datatype will be changed accordingly
# 
# - term to integer
# 
# - annual_inc to integer
# 
# #### loan_status is the target feature with categorical values, therefore feature engineering will be performed.

# ### **Data Preprocessing Part1**

# In[6]:


#data clean
df_clean = df.copy()


# #### Handling Missing Values

# In[7]:


# Total null values
total_null = df_clean.isnull().sum()
percent_missing = df_clean.isnull().sum() * 100/ len(df)
dtypes = [df_clean[col].dtype for col in df_clean.columns]
df_missing_value = pd.DataFrame({'total_null': total_null,
                                'data_type': dtypes,
                                'percent_missing': percent_missing})
df_missing_value.sort_values('percent_missing', ascending = False,inplace = True)
missing_value = df_missing_value[df_missing_value['percent_missing']>0].reset_index()
missing_value


# In[8]:


# Drop feature that have more than 50% missing value
col_full_null = df_missing_value.loc[df_missing_value['percent_missing']> 50].index.tolist()
df_clean.drop(columns=col_full_null, inplace = True)

# Drop unrelevant feature
df_clean.drop(['policy_code','application_type','title', 'url','zip_code', 'addr_state','sub_grade','Unnamed: 0','id','member_id','emp_title','pymnt_plan','issue_d'], axis = 1, inplace=True)


# ### **Feature Engineering**

# #### Target of Feature

# In[9]:


# target of feature is loan status 
df_clean['loan_status'].value_counts()


# - There are 9 unique values in the loan_status column that will be the target model.
# - Divided into two groups such as binary classification, namely "suitable" with the number 1 and "not suitable" with the number 0
# - suitable is defined as having a loan status of Current, Fully Paid , and In Grace Period
# - bad loan is defined as having a loan status other than suitable

# In[10]:


# listing values that are into suitable for loan approval list
suitable = ['Current', 'Fully Paid', 'In Grace Period']

# encoding loan_status
df_clean['loan_status'] = np.where(df_clean['loan_status'].isin(suitable), 1, 0)
df_clean['loan_status'].value_counts()/len(df_clean)*100


# #### Date-Time Features

# In[11]:


# format data type
df_clean['earliest_cr_line'] = pd.to_datetime(df_clean['earliest_cr_line'], format = '%b-%y')
df_clean['last_credit_pull_d'] = pd.to_datetime(df_clean['last_credit_pull_d'], format = '%b-%y')
df_clean['last_pymnt_d'] = pd.to_datetime(df_clean['last_pymnt_d'], format = '%b-%y')
df_clean['next_pymnt_d'] = pd.to_datetime(df_clean['next_pymnt_d'], format = '%b-%y')


# In[12]:


# count the distance until the assumed date (March 1st, 2016)
df_clean['earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2016-03-01') - df_clean['earliest_cr_line']) / np.timedelta64(1, 'M')))
df_clean['last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2016-03-01') - df_clean['last_credit_pull_d']) / np.timedelta64(1, 'M')))
df_clean['last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2016-03-01') - df_clean['last_pymnt_d']) / np.timedelta64(1, 'M')))
df_clean['next_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2016-03-01') - df_clean['next_pymnt_d']) / np.timedelta64(1, 'M')))


# #### Change Datatype

# In[13]:


#term to integer
df_clean['term'] = df_clean['term'].apply(lambda term: int(term[:3])) # filter for first 2 character


# ### **Exploratory Data Analysis**

# #### Statistical Analysis

# In[14]:


# Statistic Analysis for numerical features
df_clean.describe().T       


# In[15]:


# Categorical Statistic value
df_clean.describe(include = 'O').T


# #### Multivariate Analysis

# In[16]:


plt.figure(figsize=(22,22))
sns.heatmap(df_clean.corr(), cmap='GnBu', annot=True, fmt='.2f')
plt.show()


# From the correlation heatmap, there are features with highly correlation values to the loan_status. To prevent multicolinearity, some of these features will be dropped using threshold value 0.7.

# ### **Data Preprocessing Part2**

# #### Drop same features 

# In[17]:




# create a square matrix with dimensions equal to the number of features
cor_matrix = df_clean.corr().abs()

# select the upper triangular. nb: choosing upper or lower will have the same result
upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))

# create drop list with 0.7 as threshold
hicorr_drop_list = [column for column in upper.columns if any(upper[column] > 0.7)]

# show drop_list
hicorr_drop_list


# In[18]:


# drop
df_clean.drop(hicorr_drop_list, axis = 1, inplace=True)


# In[19]:


df_clean.corr()['loan_status'].sort_values()


# #### Handling Missing Values

# In[20]:


# Feature `tot_coll_amt`,`tot_cur_bal`,`total_rev_hi_lim` replace missing value with "0" because asumption that customer didn't borrowed again
for col in ['tot_coll_amt','tot_cur_bal']:
    df_clean[col] = df_clean[col].fillna(0)
    
# Numerical columns replace missing value with "Median"
for col in df_clean.select_dtypes(exclude = 'object'):
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
df_clean.isnull().sum()

# Categorical columns replace missing value with "Mode"
for col in df_clean.select_dtypes(include = 'object'):
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
df_clean.isnull().sum()


# In[21]:


## check 
df_clean.info()


# #### Business Insight

# In[22]:


df_eda = df.copy()


# In[23]:


# create target feature
df_eda['risk'] = np.where((df_eda['loan_status'] =='Charged Off') | 
                         (df_eda['loan_status'] =='Default') | 
                         (df_eda['loan_status'] =='Late (31-120 days)') | 
                         (df_eda['loan_status'] =='Late (16-30 days)') | 
                         (df_eda['loan_status'] =='Does not meet the credit policy. Status:Charged Off'),'Bad Risk','Good Risk')


# ### **The Number of Applicants by Loan Status**

# In[24]:


# table
loan_grp = df_eda.groupby('loan_status').size().reset_index()
loan_grp.columns = ['target','total']
loan_grp['%'] = round(loan_grp['total']*100/sum(loan_grp['total']),2)
loan_grp.sort_values(by='total', ascending=False).style.background_gradient(cmap='Blues')


# In[25]:


# visualization
sns.set_style('white')
fig = plt.figure(figsize = (8,5))
grp = df_eda['loan_status'].value_counts().sort_values(ascending=True)
grp.plot(kind='barh', color='blue', width=0.8)
plt.title('The Number of Applicants based on Loan Status\n', fontsize=16)
plt.ylabel('Status of Loan')
plt.xlabel('Count of Applicants')
plt.show()


# ##### **There are about 48% which is equal to about 224,226 applicants with loan status of Current, followed by loan status of Fully Paid with 39.6% or equal to 184,739 applicants.**

# In[26]:


grp_risk = df_eda.groupby('risk').size().reset_index()
grp_risk.columns = ['target','total']
grp_risk['%'] = round(grp_risk['total']*100/sum(grp_risk['total']),2)
grp_risk.sort_values(by='total', ascending=False)


# In[27]:


# visualization
sns.set_style('whitegrid')
labels = ['Bad Risk', 'Good Risk']
colors =  ["black", "blue"]              #["#ba3d51", "#68abad"]
sns.set_palette(sns.color_palette(colors))

fig, ax = plt.subplots(figsize=(6, 6))

patches, texts, pcts = plt.pie(grp_risk['total'], labels=labels, autopct='%.2f%%', 
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'fontsize': 13})

# for each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='white', fontweight=800)
plt.setp(texts, fontweight=800)
ax.set_title('Loan Status', fontsize=14, fontweight='bold')
plt.tight_layout()


# ##### **It is observed that this dataset is highly imbalanced with the 11% minority class, i.e Bad Risk and 88% majority class, i.e Good Risk.**

# ### **Risk Status by Term**

# In[45]:


# visualization
plt.figure(figsize=(10,5))
sns.set_style('whitegrid')

fig = sns.countplot(data = df_eda, x='term', hue = 'risk', palette='Set1')
plt.title('Risk Status by Term\n', fontsize=12)
plt.xlabel('\nTerm', fontsize=12)


# - Loan term tell us about the number of payments on the loan.
# - There are only two types of loan terms, either 36 months or 60 months. Most of the loans (73%) are shorter, with a term of 36 months.
# - Loans with 36 months period are almost twice as likely to bad risk as loans with 60 months period.

# ### **Risk Status by Home Ownership**

# In[48]:


## check 
df_eda['home_ownership'].unique()


# In[49]:


# reduce the number of categories of home ownership
def func(row):
    if row['home_ownership'] == 'MORTGAGE':
        val = 'MORTGAGE'
    elif (row['home_ownership'] == 'RENT'):
        val ='RENT'
    elif (row['home_ownership'] == 'OWN'):
        val ='OWN'
    else:
        val ='OTHERS'
    return val

df_eda['home_ownership'] = df_eda.apply(func, axis=1)


# In[50]:


# visualization
plt.figure(figsize=(10,5))
sns.set_style('whitegrid')

fig = sns.countplot(data = df_eda, x='home_ownership', hue = 'risk', palette='GnBu')
plt.title('Risk Status by Home Ownership\n', fontsize=12)
plt.xlabel('\nHome Ownership', fontsize=12)


# - The home ownership feature is category provided by the applicant's during registration.
# - Most applicants have an existing mortgage (50%) or are currently renting a home (40%).
# - Applicants who have an existing mortgage or are currently renting a home have a higher probability of bad risk.

# ### **Loan Status by purpose**

# In[51]:


# reduce the number of categories of purpose
df_eda['purpose'] = np.select([(df_eda['purpose'] == 'debt_consolidation'),
                               (df_eda['purpose'] == 'credit_card'),
                               (df_eda['purpose'] == 'other'),
                               (df_eda['purpose'] == 'major_purchase'),
                               (df_eda['purpose'].str.contains('home|car|house')),
                               (df_eda['purpose'].str.contains('small|medic|moving|vaca|wedd|educa|renew'))],
                               ['debt_consolidation','credit_card','other','major_purchase','Object_spending','Life_spending'])


# In[52]:


# Bad Risk by Purpose
df_pr = df_eda[(df_eda['risk'] == 'Bad Risk')]
df_pr = df_pr.groupby(['purpose'])['member_id'].agg(['count']).reset_index()
df_pr.columns = ['Reason', 'Bad Risk']
df_pr['percentage'] = round((df_pr['Bad Risk']/len(df_clean))*100,3)
df_pr = df_pr.sort_values('Bad Risk', ascending=False).reset_index(drop=True)
df_pr


# In[53]:


# visualization
plt.figure(figsize=(10,5))
sns.set_style('whitegrid')

fig = sns.barplot(data = df_pr, x='Reason', y='Bad Risk', palette='GnBu')
plt.title('Bad Risk by Purpose\n', fontsize=12)
plt.xlabel('\nPurpose', fontsize=12)


# - Most of Customer that type Bad Risk is in debt_consolidation 6.8%
# - For the second is credit_card 
# - This shows that for debt reasons, the customer is more likely to be the type of customer that is Bad 

# ## **Feature Selection** 

# In[54]:


df_clean.columns.tolist()


# In[55]:


df_fs = df_clean.copy()


# In[56]:


# code automation

def woe(raw, feature_name):
    # probability analysis
    feature_name = raw.groupby(feature_name).agg(num_observation=('loan_status','count'),
                                                good_loan_prob=('loan_status','mean')).reset_index()
    
    # find the feature proportion
    feature_name['feat_proportion'] = feature_name['num_observation']/(feature_name['num_observation'].sum())
    
    # find number of approved loan behavior
    feature_name['num_loan_approve'] = feature_name['feat_proportion'] * feature_name['num_observation']

    # find number of declined loan behavior
    feature_name['num_loan_decline'] = (1-feature_name['feat_proportion']) * feature_name['num_observation']

    # find approved loan proportion
    feature_name['prop_loan_approve'] = feature_name['num_loan_approve'] / (feature_name['num_loan_approve'].sum())

    # find declined loan proportion
    feature_name['prop_loan_decline'] = feature_name['num_loan_decline'] / (feature_name['num_loan_decline'].sum())

    # calculate weight of evidence
    feature_name['weight_of_evidence'] = np.log(feature_name['prop_loan_approve'] / feature_name['prop_loan_decline'])

    # sort values by weight of evidence
    feature_name = feature_name.sort_values('weight_of_evidence').reset_index(drop=True)
    
    # calculate information value
    feature_name['information_value'] = (feature_name['prop_loan_approve']-feature_name['prop_loan_decline']) * feature_name['weight_of_evidence']
    feature_name['information_value'] = feature_name['information_value'].sum()

    #Show
    feature_name = feature_name.drop(['feat_proportion','num_loan_approve','num_loan_decline','prop_loan_approve','prop_loan_decline'],axis = 1)

    return feature_name


# ### **Catagorical Features**

# In[57]:


# Categorical Statistic Value
df_clean.describe(include = 'O').T


# #### Grade

# In[58]:


##grade
woe(df_fs,'grade')


# ### emp_length

# In[59]:


woe(df_fs,'emp_length')


# #### home_ownership

# In[60]:


df_fs['home_ownership'] = np.where(df_fs['home_ownership']=='ANY','OTHER',
                       np.where(df_fs['home_ownership']=='NONE','OTHER',df_fs['home_ownership']))


# In[61]:


woe(df_fs,'home_ownership')


# #### verification_status

# In[62]:


woe(df_fs,'verification_status')


# #### purpose

# In[63]:


woe(df_fs,'purpose')


# #### initial_list_status

# In[64]:


woe(df_fs,'initial_list_status')


# ### **Numerical Features**

# In[65]:


df_clean.describe().T


# #### loan_amnt

# In[66]:


# refining class = 10 class
df_fs['loan_amnt_fs'] = pd.cut(df_fs['loan_amnt'], 10)
woe(df_fs,'loan_amnt_fs')


# #### term

# In[67]:


woe(df_fs,'term')


# #### int_rate

# In[68]:


# refining class = 10 class
df_fs['int_rate_fs'] = pd.cut(df_fs['int_rate'], 10)
woe(df_fs,'int_rate_fs')


# #### annual_inc

# In[69]:


# refining class = 4 class
df_fs['annual_inc_fs'] = pd.cut(df_fs['annual_inc'], 4)
woe(df_fs,'annual_inc_fs')


# #### dti

# In[70]:


# refining class = 10 class
df_fs['dti_fs'] = pd.cut(df_fs['dti'], 10)
woe(df_fs,'dti_fs')


# #### delinq_2yr

# In[71]:


# this feature will be encoded, if values = 0 return 0, if its greater than 0 return 1, if > 5, return 2
df_fs['delinq_2yrs_fs'] = np.where(df_fs['delinq_2yrs'] > 3, 3,
                                 np.where(df_fs['delinq_2yrs'] == 2, 2,
                                 np.where(df_fs['delinq_2yrs'] == 1,1,0)))

# show
woe(df_fs,'delinq_2yrs_fs')


# #### earliest_cr_line

# In[72]:


# refining class = 10 class
df_fs['earliest_cr_line_fs'] = pd.cut(df_fs['earliest_cr_line'], 10)
woe(df_fs,'earliest_cr_line_fs')


# #### inq_last_6mths

# In[73]:


#encoding the feature
df_fs['inq_last_6mths_fs'] = np.where(df_fs['inq_last_6mths'] == 0,0,
                                    np.where((df_fs['inq_last_6mths'] > 0)&(df_fs['inq_last_6mths'] <=3),1,
                                    np.where((df_fs['inq_last_6mths']>3)&(df_fs['inq_last_6mths']<=6),2,
                                    np.where((df_fs['inq_last_6mths']>6)&(df_fs['inq_last_6mths']<=9),3,4))))

# show
woe(df_fs,'inq_last_6mths_fs')


# #### open_acc

# In[74]:


# refining class = 5 class
df_fs['open_acc_fs'] = pd.cut(df_fs['open_acc'], 5)
woe(df_fs,'open_acc_fs')


# #### pub_rec

# In[75]:


# refining class = 5 class
df_fs['pub_rec_fs'] = pd.cut(df_fs['pub_rec'], 5)
woe(df_fs,'pub_rec_fs')


# #### revol_bal

# In[76]:


#encode to new class
df_fs['revol_bal_fs'] = np.where((df_fs['revol_bal']>=0)&(df_fs['revol_bal']<=5000),0,
                               np.where((df_fs['revol_bal']>5000)&(df_fs['revol_bal']<=10000),1,
                               np.where((df_fs['revol_bal']>10000)&(df_fs['revol_bal']<=15000),2,3)))

# show
woe(df_fs,'revol_bal_fs')


# #### revol_util

# In[77]:


#encoding into new class
df_fs['revol_util_fs'] = np.where((df_fs['revol_util']>=0)&(df_fs['revol_util']<=20),0,
                                np.where((df_fs['revol_util']>20)&(df_fs['revol_util']<=40),1,
                                np.where((df_fs['revol_util']>40)&(df_fs['revol_util']<=60),2,
                                np.where((df_fs['revol_util']>60)&(df_fs['revol_util']<=80),3,4))))

# show
woe(df_fs,'revol_util_fs')


# #### total_acc

# In[78]:


# refining class = 5 class
df_fs['total_acc_fs'] = pd.cut(df_fs['total_acc'], 5)
woe(df_fs,'total_acc_fs')


# #### out_prncp

# In[79]:


#Encoding into new class
df_fs['out_prncp_fs'] = np.where((df_fs['out_prncp']>=0)&(df_fs['out_prncp']<=1000),0,
                               np.where((df_fs['out_prncp']>1000)&(df_fs['out_prncp']<=10000),1,
                               np.where((df_fs['out_prncp']>10000)&(df_fs['out_prncp']<=17000),2,3)))

# show
woe(df_fs,'out_prncp_fs')


# #### total_rec_late_fee

# In[80]:


#Encoding into new class
df_fs['total_rec_late_fee_fs'] = np.where(df_fs['total_rec_late_fee']==0,0,1)

# show
woe(df_fs,'total_rec_late_fee_fs')


# #### recoveries

# In[81]:


# refining class = 5 class
df_fs['recoveries_fs'] = pd.cut(df_fs['recoveries'], 5)
woe(df_fs,'recoveries_fs')


# #### last_pymnt_d

# In[82]:


#Encoding into new class
df_fs['last_pymnt_d_fs'] = np.where(df_fs['last_pymnt_d']==2,0,
                                  np.where((df_fs['last_pymnt_d']>2)&(df_fs['last_pymnt_d']<=4),1,
                                  np.where((df_fs['last_pymnt_d']>4)&(df_fs['last_pymnt_d']<=6),2,
                                  np.where((df_fs['last_pymnt_d']>6)&(df_fs['last_pymnt_d']<=12),3,4))))
                                           
# show
woe(df_fs,'last_pymnt_d_fs')


# #### collections_12_mths_ex_med

# In[83]:


# refining class = 5 class
df_fs['collections_12_mths_ex_med_fs'] = pd.cut(df_fs['collections_12_mths_ex_med'], 5)
woe(df_fs,'collections_12_mths_ex_med_fs')


# #### acc_now_delinq

# In[84]:


# refining class = 5 class
df_fs['acc_now_delinq_fs'] = pd.cut(df_fs['acc_now_delinq'], 5)
woe(df_fs,'acc_now_delinq_fs')


# #### tot_coll_amt

# In[85]:


# refining class = 5 class
df_fs['tot_coll_amt_fs'] = pd.cut(df_fs['tot_coll_amt'], 5)
woe(df_fs,'tot_coll_amt_fs')


# #### tot_cur_bal

# In[86]:


# refining class = 5 class
df_fs['tot_cur_bal_fs'] = pd.cut(df_fs['tot_cur_bal'], 5)
woe(df_fs,'tot_cur_bal_fs')


# ### **Summary**

# Feature we will drop because have:
# 
# - Information value <0.02 (useless predictive)
# - Information value > 0.5 (suspicious predictive)
# - Feature that not make sense to bin

# In[87]:


drop_list = ['emp_length','verification_status', 'purpose', 'term','annual_inc', 'delinq_2yrs','earliest_cr_line', 'total_acc','open_acc', 'pub_rec', 
             'out_prncp', 'total_rec_late_fee','recoveries', 'collections_12_mths_ex_med', 
             'acc_now_delinq','tot_coll_amt','tot_cur_bal']

print('number of features that we will drop :',len(drop_list))


# ### **Feature Encoding**

# In[88]:


df_test = df_clean.copy()


# In[89]:


# drop unused features
df_test.drop(['emp_length','verification_status', 'purpose', 'term','annual_inc', 'delinq_2yrs','earliest_cr_line', 'total_acc','open_acc', 'pub_rec', 
             'out_prncp', 'total_rec_late_fee','recoveries', 'collections_12_mths_ex_med', 
             'acc_now_delinq','tot_coll_amt','tot_cur_bal'], axis = 1, inplace=True)


# In[90]:


cat = df_test.select_dtypes(include='O').columns
num = df_test.select_dtypes(exclude='O').columns


# ### **Cat Feature Encoding**

# In[91]:


# handle with one hot encoding
for cat in ['grade', 'home_ownership', 'initial_list_status']:
  onehots = pd.get_dummies(df_clean[cat], prefix=cat)
  df_test = df_test.join(onehots)


# In[92]:


df_test = df_test.drop(columns=['grade', 'home_ownership', 'initial_list_status'], axis =1)
df_test.head()


# In[93]:


#check
df_test.shape


# ### **Num Feature Encoding**

# In[94]:


from sklearn.preprocessing import LabelEncoder


# In[95]:


#Label encoding

le = LabelEncoder()

columns = [ 'loan_amnt', 'int_rate', 'loan_status', 'dti', 'inq_last_6mths',
       'revol_bal', 'revol_util', 'last_pymnt_d'
]

for col in columns:
    df_test[col] = le.fit_transform(df_clean[col])


# In[96]:


df_test.var()


# In[97]:


#Check
# Show Data Rows & Features Number
print("Total Rows :", df_test.shape[0])
print("Total Features :", df_test.shape[1])


# ## **Modelling**

# In[98]:


X = df_test.drop(labels=['loan_status'],axis=1)
y = df_test[['loan_status']]


# ### **Split Dataset**

# In[99]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify=y,random_state = 42)


# In[100]:


from datetime import datetime as dt
from collections import defaultdict
from xgboost import XGBClassifier
import lightgbm as lgb
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# ### **Summary of Modeling ML**

# In[1]:


def modelling(x_train,x_test,y_train,y_test):
    result = defaultdict(list)
    
    knn = KNeighborsClassifier()
    xgb = XGBClassifier()
    rf = RandomForestClassifier()
    grad = GradientBoostingClassifier()
    LGBM = lgb.LGBMClassifier()

    
    list_model = [('K-Nearest Neighbor',knn),
                  ('XgBoost', xgb),
                 ('Random Forest',rf),
                  ('Gradient Boosting',grad),
                  ('LightGBM',LGBM)
                  ]

    for model_name, model in list_model:
        model.fit(x_train,y_train)
        y_pred_proba = model.predict_proba(x_test)

        y_pred = model.predict(x_test)
        
        accuracy = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        AUC = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        result['model_name'].append(model_name)
        result['model'].append(model)
        result['accuracy'].append(accuracy)
        result['recall'].append(recall)
        result['precision'].append(precision)
        result['AUC'].append(AUC)
        
   return result


# In[2]:


evaluation_summary = modelling(X_train,X_test,y_train,y_test)
evaluation_summary = pd.DataFrame(evaluation_summary)
evaluation_summary


# ### **Confusion Matric**

# In[ ]:


lg_model = lgb.LGBMClassifier()
lg_model.fit(X_train, y_train)
y_pred_lg = lg_model.predict(X_test)


# In[ ]:


def show_cmatrix(ytest, pred):
    # Creating confusion matrix 
    cm = confusion_matrix(ytest, pred)

    # Putting the matrix a dataframe form  
    cm_df = pd.DataFrame(cm, index=['Bad Loan', 'Good Loan'],
                 columns=['Predicted Bad Loan', 'Predicted Good Loan'])
    
    # visualizing the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(9,5))
        
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues",xticklabels=cm_df.columns, yticklabels=cm_df.index, annot_kws={"size": 20})
    plt.title("Confusion Matrix", size=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label');


# In[ ]:


show_cmatrix(y_test, y_pred_lg)


# ### **Feature Importence**

# In[ ]:


# plt.figsize(10, 8)
feat_importances = pd.Series(lg_model.feature_importances_, index=X.columns)
ax = feat_importances.nlargest(10).plot(kind='barh', figsize=(10, 8),color='turquoise')
ax.set(facecolor = "white")
ax.invert_yaxis()


# ### **Simulation with No-Model and Model**

# In[ ]:


data_benefit = df_clean.copy()


# In[ ]:


#define values
ambiguous = ['Current', 'In Grace Period']
good_loan =  ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']

#drop rows that contain ambiguous ending
data_benefit = data_benefit[data_benefit.loan_status.isin(ambiguous) == False]

#create new column to classify ending when 1 equal bad loan, and 0 equal good loan
data_benefit['loan_ending'] = np.where(data_benefit['loan_status'].isin(good_loan), 0, 1)


# In[ ]:


#Growth Calculation 
Total = data_benefit['loan_ending'].count()
Bad =  data_benefit[data_benefit['loan_ending']==1]['loan_ending'].count()
Good =  data_benefit[data_benefit['loan_ending']==0]['loan_ending'].count()
PredRate = 0.889
PredBad = round(Bad*PredRate)
PredGood = Bad-PredBad

print('----- Existing -----')
print('\t\t\t','count', 'percentage')
print('Total_loan : \t\t', Total)
print('Bad_loan : \t\t', Bad, ',', round(Bad/Total*100,1),'%')
print('Good_loan : \t\t', Good, ',', round(Good/Total*100,1),'%')
print()
print('----- After Modeling -----')
print('\t\t\t','count', 'percentage')
print('Total_loan : \t\t', Total)
print('Bad_loan : \t\t', Bad, ',', round(Bad/Total*100,1),'%')
print('  Predicted Bad_loan : \t', round(PredBad), ',', round(PredBad/Bad*100,1),'%')
print('  Predicted Good_loan : ', round(PredGood), ',', round(PredGood/Bad*100,1),'%')
print('Bad_loan After Pred : \t', Bad-PredBad, ',', round((Bad-PredBad)/Total*100,1),'%')
print('Bad_loan Growth rate : \t', round(((Bad-PredBad)/(Bad)-1)*100,1), '%')
print('Good_loan : \t\t', Good, ',', round(Good/Total*100,1),'%')
print('Good_loan After Pred : \t', Good+PredBad, ',', round((Good+PredBad)/Total*100,1),'%')
print('Good_loan Growth rate : \t', round(((Good+PredBad)/(Good)-1)*100,1), '%')

