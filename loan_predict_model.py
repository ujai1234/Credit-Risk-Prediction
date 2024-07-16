#!/usr/bin/env python
# coding: utf-8

# ## 1. Business Understanding

# 1. This project comes from a lending company
# 2. Objective: to build a model that can predict credit risk using a dataset provided by the company consisting of accepted and rejected loan data.
# 3. create visual media presenting solutions to client with PPT

# ## 2. Analytic Approach

# The problem given by the lending company is the risk of lending credit. the lending company wants to create a model that can predict credit risk to predict customers can borrow or be rejected. the analysis approach taken is a 'predictive model'.

# ## 3. Data Requirements

# The data required is the accepted and rejected loan data and the authorization of each data feature. 

# ## 4. Data Collection

# ##### Import Library

# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[137]:


df = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)
df.head()


# ## 5. Data Understanding

# In[138]:


df.shape


# In[139]:


df.info()


# In[140]:


df.drop(columns='Unnamed: 0', inplace=True)
df.columns, df.shape


# In[141]:


# investigate target column
df1 = df.copy()
df1.loan_status.unique()


# In[142]:


# 
df1['target'] = np.where(df1.loc[:, 'loan_status'].isin(['Charged Off', 
                                                         'Default',  
                                                         'Late (31-120 days)', 
                                                         'Late (16-30 days)', 
                                                         'Does not meet the credit policy. Status:Charged Off'])
                                                         , 1, 0)


# <h4><p><b>at this stage we divide the target into two, namely good debt = 1 and bad debt = 0 based on the loan status.</b></p></h>
# 
# <h5><p><b>good borrowers:</b>
# <p>-Fully Paid
# <p>-Current
# <p>-In Grace Period
# <p>Does not meet the credit policy. Status:Fully Paid</p>
# 
# <p><b>bad borrowers:</b>
# <p>-Charged Off
# <p>-Default
# <p>-Late (31-120 days)
# <p>-Late (16-30 days)
# <p>-Does not meet the credit policy. Status:Charged Off</p></h>

# In[143]:


df1.target.value_counts()


# In[144]:


df1.groupby('target').agg({'funded_amnt':'sum', 'total_pymnt':'sum'}).reset_index()


# The data above shows that the company suffered a loss from bad debt of $760916150

# In[145]:


# Counts the number of null values in each column of the frame data and sort based on null count
df_drop = df1.isnull().sum().sort_values()
# get data from df1
df_drop = df_drop[df_drop == df1.shape[0]]
# drop null based on index
df_drop = list(df_drop.index)
print(df_drop)


# #### Drop null feature columns

# In[146]:


df1.drop(columns=df_drop, inplace=True)
print(f"Data dimension before drop: {df.shape}")
print(f"Data dimension after drop: {df1.shape}")


# ### Summary of Statistic Descriptive for Categorical and Numerical Feature

# ##### Separating numerical and categorical features

# In[147]:


# create an empty list 
numeric = []
categorical = []

for i in df1.columns:
    if df1[i].dtype == 'object':
        categorical.append(i)
    else:
        numeric.append(i)
print(f"The amount of numerical feature is: {len(numeric)}")
print(f"The amount of categorical feature is: {len(categorical)}")


# #### Summary of Statistic Descriptive for numercal features

# In[148]:


df1[numeric].describe().transpose()


# In[149]:


# investigate numerical column
df1[numeric]['policy_code'].unique()


# In[150]:


# drop columns that are not useful for model
df1.drop(columns=['policy_code', 'id', 'member_id'], inplace=True)


# In[151]:


print(f"Data dimension after drop: {df1.shape}")


# #### Summary of Statistic Descriptive for categorical features

# In[152]:


df1[categorical].describe().transpose()


# In[153]:


# drop columns that are not useful for model
df1.drop(columns=['application_type', 'url', 'desc', 
                  'title', 'addr_state', 'zip_code', 'emp_title'], 
                  inplace=True)


# In[154]:


print(f"Data dimension after drop: {df1.shape}")


# ## 🔍 Exploratory Data Analysis

# #### Univariate Analysis: Numerical Features
# 1. Summary statistics 

# In[155]:


df_an = df1.copy()


# In[156]:


df_an["funded_amnt"].value_counts()


# In[157]:


df_an["funded_amnt_inv"].value_counts()


# In[158]:


# create a new empty list 
numeric1 = []
categorical1 = []

for i in df_an.columns:
    if df_an[i].dtype == 'object':
        categorical1.append(i)
    else:
        numeric1.append(i)


# In[159]:


df_an[numeric1].describe().transpose()


# 2. Chart analysis

# In[160]:


print(numeric1)


# In[161]:


num = ['funded_amnt','funded_amnt_inv', 'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 
       'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 
       'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
       'last_pymnt_amnt', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 
       'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']


# In[162]:


def plot_hist(variable):
    plt.figure(figsize=(6,3))
    plt.hist(df_an[variable], bins = 30)
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Graphic of {variable}")
    plt.xticks(rotation = 30)
    plt.show()

for i in num:
    plot_hist(i) 


# In[163]:


df_an[categorical1].describe().transpose()


# #### Univariate Analysis: Categorical Features

# Feature engineering for Date analysis

# In[164]:


date = []
for i in categorical1:
    if df_an[i].nunique() > 35:
        date.append(i)
print(date)


# In[165]:


df_date = df_an[date]
df_date.head()


# In[166]:


df_date = df_date.drop(['funded_amnt', 'funded_amnt_inv'], axis=1)
df_date.head()


# Separate month and year features

# In[167]:


df_date['issue_d_month'] = df_date.issue_d.str[:3]
df_date['issue_d_year'] = np.where(df_date.issue_d.str[4:].astype('float64')>20,'19'+ df_date.issue_d.str[4:],'20'+ df_date.issue_d.str[4:])

df_date['earliest_cr_line_month'] = df_date.earliest_cr_line.str[:3]
df_date['earliest_cr_line_year'] = np.where(df_date.earliest_cr_line.str[4:].astype('float64')>20,'19'+ df_date.earliest_cr_line.str[4:],'20'+ df_date.earliest_cr_line.str[4:])

df_date['last_pymnt_d_month'] = df_date.last_pymnt_d.str[:3]
df_date['last_pymnt_d_year'] = np.where(df_date.last_pymnt_d.str[4:].astype('float64')>20,'19'+ df_date.last_pymnt_d.str[4:],'20'+ df_date.last_pymnt_d.str[4:])

df_date['next_pymnt_d_month'] = df_date.next_pymnt_d.str[:3]
df_date['next_pymnt_d_year'] = np.where(df_date.next_pymnt_d.str[4:].astype('float64')>20,'19'+ df_date.next_pymnt_d.str[4:],'20'+ df_date.next_pymnt_d.str[4:])

df_date['last_credit_pull_d_month'] = df_date.last_credit_pull_d.str[:3]
df_date['last_credit_pull_d_year'] = np.where(df_date.last_credit_pull_d.str[4:].astype('float64')>20,'19'+ df_date.last_credit_pull_d.str[4:],'20'+ df_date.last_credit_pull_d.str[4:])


# In[168]:


df_date.head()


# In[169]:


df_date = df_date.drop(['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d'], axis=1)
df_date.head()


# In[170]:


#df_date.isnull().sum()


# In[171]:


# df_date = df_date.dropna()
# df_date.head()


# #### issue_d_year features

# In[172]:


plt.figure(figsize=(8,5))
sns.countplot(df_date, x=df_date.issue_d_year.sort_values())
plt.xticks(rotation=90)
plt.show()


# The most funded loan years were in 2014, meaning that the number of loans approved is increasing every year.

# #### earliest_cr_line_year features

# In[173]:


plt.figure(figsize=(10,5))
sns.countplot(df_date, x=df_date.earliest_cr_line_year.sort_values())
plt.xticks(rotation=90)
plt.show()


# The month in which the borrower first opened the reported credit limit is skew negatively distributed 

# #### last_pymnt_d_year features

# In[174]:


plt.figure(figsize=(8,5))
sns.countplot(df_date, x=df_date.last_pymnt_d_year.sort_values())
plt.xticks(rotation=90)
plt.show()


# Most annual payments received in 2016

# #### next_pymnt_d_year features

# In[175]:


plt.figure(figsize=(8,5))
sns.countplot(df_date, x=df_date.next_pymnt_d_year.sort_values())
plt.xticks(rotation=90)
plt.show()


# The next scheduled payment date is dominated by 2016

# #### last_credit_pull_d_year features

# In[176]:


plt.figure(figsize=(8,5))
sns.countplot(df_date, x=df_date.last_credit_pull_d_year.sort_values())
plt.xticks(rotation=90)
plt.show()


# The last credit pull year payment is dominated by 2016

# #### loan_status feature

# In[177]:


df_cat = df_an.copy()


# In[178]:


plt.figure(figsize=(8,5))
sns.countplot(df_cat, x=df_an.loan_status.sort_values())
plt.xticks(rotation=90)
plt.show()


# most loan statuses are current and fully paid, which means that there are more good borrowers (paying) than bad ones (not paying). 

# #### empth_length features

# In[179]:


plt.figure(figsize=(8,5))
sns.countplot(df_cat, x=df_an.emp_length.sort_values())
plt.xticks(rotation=90)
plt.show()


# Most borrowers' years of service are > 10 years, while borrowers' years of service < 1 year are ranked second.

# #### grade and sub_grade feature

# In[180]:


df_cat['grade'] = df_cat['grade'].sort_values()
df_cat['sub_grade'] = df_cat['sub_grade'].sort_values()


# In[181]:


df_grad = ['grade', 'sub_grade']

for col in df_grad:
    # Count the number of occurrences and convert to DataFrame
    value_counts = df_cat[col].value_counts().reset_index()  
    # Renaming columns
    value_counts = value_counts.rename(columns={col: 'count', 'index': col})  
    # Sort DataFrame by 'count' column
    value_counts = value_counts.sort_values(by='count', ascending=False)

    # Draw a countplot with the sorted order
    fig = sns.barplot(data=value_counts, x=col, y='count')
    plt.xticks(rotation=90)
    plt.show()


# The grade of the most borrowers is B, C and D, meaning that the average borrower grade is a good borrower. for details of its classification can be seen in sub_grade

# #### Term feature

# In[182]:


plt.figure(figsize=(8,5))
sns.countplot(df_cat, x=df_an.term.sort_values())
plt.xticks(rotation=90)
plt.show()


# #### pymnt_plan feature

# In[183]:


plt.figure(figsize=(8,5))
sns.countplot(df_cat, x=df_an.pymnt_plan.sort_values())
plt.xticks(rotation=90)
plt.show()


# #### home_ownership feature

# In[184]:


plt.figure(figsize=(8,5))
sorted_home_ownership = df_an['home_ownership'].value_counts().index
sns.countplot(data=df_an, x='home_ownership', order=sorted_home_ownership)
plt.xticks(rotation=90)
plt.show()


# #### purpose feature

# In[185]:


plt.figure(figsize=(8,5))
sorted_purpose = df_an['purpose'].value_counts().index
sns.countplot(data=df_cat, x='purpose', order=sorted_purpose)
plt.xticks(rotation=90)
plt.show()


# #### initial_list_status feature

# In[186]:


plt.figure(figsize=(8,5))
sns.countplot(df_cat, x=df_an.initial_list_status.sort_values())
plt.xticks(rotation=90)
plt.show()


# In[187]:


import hvplot.pandas


# ### Bivariate Analysis : Categorical Features

# #### loan_amnt & installment
# * installment: The monthly payment owed by the borrower if the loan originates.
# * loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.

# In[188]:


# Then you can create a visualization by using hvplot
installment = df_an.hvplot.hist(
    y='installment', by='target', subplots=False, 
    width=350, height=400, bins=50, alpha=0.4, 
    title="Installment by Loan Status", 
    xlabel='Installment', ylabel='Counts', legend='top'
)

loan_amnt = df_an.hvplot.hist(
    y='loan_amnt', by='target', subplots=False, 
    width=350, height=400, bins=30, alpha=0.4, 
    title="Loan Amount by Loan Status", 
    xlabel='Loan Amount', ylabel='Counts', legend='top'
)

# Then you can combine visualizations using the '+' operator
installment + loan_amnt


# #### int_rate & annual_inc
# * int_rate: Interest Rate on the loan
# * annual_inc: The self-reported annual income provided by the borrower during registration

# In[189]:


int_rate = df_an.hvplot.hist(
    y='int_rate', by='target', alpha=0.3, width=350, height=400,
    title="Loan Status by Interest Rate", xlabel='Interest Rate', ylabel='Loans Counts', 
    legend='top'
)

annual_inc = df_an.hvplot.hist(
    y='annual_inc', by='target', bins=50, alpha=0.3, width=350, height=400,
    title="Loan Status by Annual Income", xlabel='Annual Income', ylabel='Loans Counts', 
    legend='top'
).opts(xrotation=45)

int_rate + annual_inc


# In[190]:


print(categorical)


# In[191]:


cat = ['term', 'grade','emp_length', 'home_ownership', 'verification_status',
        'pymnt_plan', 'purpose', 'initial_list_status']


# In[192]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))  # Mengatur ukuran gambar

# Mengubah tata letak subplot menjadi 2 baris dan 4 kolom
for i in range(0, len(cat)):
    plt.subplot(2, 4, i+1)
    sns.countplot(x=cat[i], data=df_cat, hue='target')
    plt.xlabel(cat[i])
    plt.xticks(rotation=90)
    plt.tight_layout()

plt.show()


# In[193]:


df_drop = df1[['target']]


# In[194]:


df_date['target'] = df_drop['target']
df_date.head()


# In[195]:


df_date.target.value_counts()


# In[196]:


years = ['issue_d_year','last_pymnt_d_year','next_pymnt_d_year','last_credit_pull_d_year']


# In[197]:


plt.figure(figsize=(20, 5))
for i in range(0, len(years)):
    plt.subplot(1, 4, i+1)
    sns.countplot(x=years[i], data=df_date, hue=df_date['target'])
    plt.xlabel(years[i])
    plt.xticks(rotation=90)
    plt.tight_layout()


# In[198]:


plt.figure(figsize=(15,5))
sns.countplot(x=df_date['earliest_cr_line_year'].sort_values(), hue=df_date['target'])
plt.xticks(rotation=90)
plt.show()


# ### Data Correlation

# In[199]:


plt.figure(figsize=(20,15))
sns.heatmap(df_an.corr(), annot=True, cmap="RdYlGn")
plt.show()


# ### Check and Handling Missing Value

# In[200]:


# Check persentase missing values untuk setiap fitur
mis_value = df_an.isnull().mean() * 100
mis_value = mis_value[mis_value > 0].reset_index()
mis_value.columns = ['feature', '%']

# Menampilkan DataFrame mis_value
print(mis_value)


# In[201]:


#filter feature yang punya null values > 40%
mis_value1 = list(mis_value['feature'][mis_value['%']>40])
mis_value2 = list(mis_value['feature'][mis_value['%']<40])

print(f"Null feature > 40%: {mis_value1} \n Null feature < 40%: {mis_value2}")


# In[202]:


# drop feature yang punya null values >40%
df_an.drop(columns=mis_value1,inplace=True)
df_an.shape


# ## Data Splitting

# In[203]:


df_an.shape


# In[204]:


from sklearn.model_selection import train_test_split


# In[205]:


X = df_an.drop('target', axis=1)
y = df_an['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y,random_state=42)


# In[206]:


y_train.value_counts(normalize=True)


# In[207]:


y_test.value_counts(normalize=True)


# ## Data Cleaning & Feature Engineering

# In[208]:


X_train.shape


# In[209]:


# Can be printed for all unique values of a column, so you can check one by one
# what unique values are dirty.

for col in X_train.select_dtypes(include= ['object','category']).columns:
    print(col)
    print(X_train[col].unique())
    print()


# In[210]:


# Columns/features that need to be cleaned
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d' 
                    ,'last_credit_pull_d']


# In[211]:


# Removing 'months' to become ''
X_train['term'].str.replace(' months', '')


# In[212]:


# Convert data type into numeric 
X_train['term'] = pd.to_numeric(X_train['term'].str.replace(' months', ''))


# In[213]:


X_train['term'].astype(int)


# In[214]:


X_train['term'].unique()


# In[215]:


X_train['term'] = X_train['term'].replace(0, 60)


# In[216]:


X_train['term'].unique()


# In[217]:


# Check what values need to be cleaned
X_train['emp_length'].unique()


# In[218]:


X_train['emp_length'] = X_train['emp_length'].str.replace('\+ years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', str(0))
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')

X_train['emp_length'].fillna(value = 0, inplace=True)
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'])


# In[219]:


X_train['emp_length']


# In[220]:


# Check Date Feature['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
col_date = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']

X_train[col_date]


# In[221]:


for column in col_date:
    mode_value = X_train[column].mode()[0]
    X_train[column].fillna(mode_value, inplace=True)

# Check if null data has been filled with mode
missing_values_after_fillna = X_train[['issue_d', 'earliest_cr_line', 'last_pymnt_d', 
                                       'last_credit_pull_d']].isnull().sum()
print("Number of Missing Values After Filling Mode:")
print(missing_values_after_fillna)


# In[222]:


X_train['earliest_cr_line'] = X_train['earliest_cr_line'].replace('26200.0', 'Oct-00')


# In[223]:


X_train[col_date] = X_train[col_date].astype(str)
X_train[col_date].dtypes


# In[224]:


X_train['issue_d'].unique()


# In[225]:


# List the dates that will be converted
# issue_d has been converted so it is not included in the list
date_conv = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']


# In[226]:


# convert date_conv to datetime format
from datetime import datetime as dt
for col in date_conv:
    X_train[col] = pd.to_datetime(X_train[col].apply(lambda x: dt.strptime(x, '%b-%y')))


# In[227]:


X_train[['issue_d','last_pymnt_d','last_credit_pull_d','earliest_cr_line']].head()


# In[228]:


X_train.drop(columns='loan_status', inplace=True)


# In[229]:


X_train.shape


# #### grad & sub_grad features

# In[230]:


X_train['grade'].unique()


# In[231]:


X_train = X_train[X_train['grade'] != 'INDIVIDUAL']


# In[232]:


X_train['grade'].unique()


# grad and sub_grad features refer to the same thing, so we will choose the grad feature because it has fewer categories.

# In[233]:


X_train.drop(columns='sub_grade', inplace=True)


# In[234]:


X_train.columns


# home_ownership cleaning

# In[235]:


X_train.home_ownership.replace({'NONE':'OTHER','ANY':'OTHER','OWN':'OTHER'},inplace=True)
X_train.home_ownership.unique()


# In[236]:


X_train['pymnt_plan'].value_counts()


# pymnt_plan has a large data imbalance so this feature will be dropped. 

# In[237]:


X_train.drop(columns='pymnt_plan', inplace=True)


# ### Repeats all steps of X_train on X_test 

# In[238]:


X_test.shape


# In[239]:


# Removing 'months' to become ''
X_test['term'].str.replace(' months', '')
# Convert data type into numeric 
X_test['term'] = pd.to_numeric(X_test['term'].str.replace(' months', ''))
X_test['term'].astype(int)
X_test['term'] = X_test['term'].replace(0, 60)

X_test['emp_length'] = X_test['emp_length'].str.replace('\+ years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', str(0))
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')

X_test['emp_length'].fillna(value = 0, inplace=True)
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'])

for column in col_date:
    mode_value = X_test[column].mode()[0]
    X_test[column].fillna(mode_value, inplace=True)

missing_values_after_fillna = X_test[['issue_d', 'earliest_cr_line', 'last_pymnt_d', 
                                       'last_credit_pull_d']].isnull().sum()
X_test['earliest_cr_line'] = X_test['earliest_cr_line'].replace('26200.0', 'Oct-00')
X_test[col_date] = X_test[col_date].astype(str)

for col in date_conv:
    X_test[col] = pd.to_datetime(X_test[col].apply(lambda x: dt.strptime(x, '%b-%y')))
    
X_test.drop(columns='loan_status', inplace=True)

X_test = X_test[X_test['grade'] != 'INDIVIDUAL']

X_test.drop(columns='sub_grade', inplace=True)

X_test.home_ownership.replace({'NONE':'OTHER','ANY':'OTHER','OWN':'OTHER'},inplace=True)
X_test.home_ownership.unique()

X_test.drop(columns='pymnt_plan', inplace=True)


# In[240]:


X_test.info()


# ### feature engineering

# In[241]:


for col in X_train.select_dtypes(include= ['object']).columns:
    print(col)
    print(X_train[col].unique())
    print()


# In[242]:


X_train['funded_amnt_inv'] = X_train['funded_amnt_inv'].str.replace('.', '', regex=False).astype(np.int64)


# In[243]:


X_train['funded_amnt'] = X_train['funded_amnt'].astype('int64')


# In[244]:


X_test['funded_amnt_inv'] = X_test['funded_amnt_inv'].str.replace('.', '', regex=False).astype(np.int64)
X_test['funded_amnt'] = X_test['funded_amnt'].astype('int64')


# #### Label Encoding

# In[245]:


X_train['earliest_cr_line'].value_counts()


# In[246]:


X_train.grade.replace({'G':0,'F':1,'E':2,'D':3,'C':4,'B':5,'A':6},inplace=True)
X_train.initial_list_status.replace({'w':0,'f':1},inplace=True)


# In[247]:


X_test.grade.replace({'G':0,'F':1,'E':2,'D':3,'C':4,'B':5,'A':6},inplace=True)
X_test.initial_list_status.replace({'w':0,'f':1},inplace=True)


# In[248]:


X_train.head()


# #### One Hot Encoding

# In[249]:


feature_onehot = ['home_ownership', 'verification_status','purpose']


# In[250]:


# One hot encoding for X_train
for col in feature_onehot :
  encode = pd.get_dummies(X_train[col], prefix=col)
  X_train = X_train.join(encode)


# In[251]:


# One hot encoding for X_test
for col in feature_onehot :
  encode = pd.get_dummies(X_test[col], prefix=col)
  X_test = X_test.join(encode)


# In[252]:


X_train.drop(columns=feature_onehot, inplace=True)
X_test.drop(columns=feature_onehot, inplace=True)


# In[253]:


X_train.tail()


# In[254]:


X_train.info()


# In[255]:


X_test.isnull().sum()


# In[256]:


X_train.isnull().sum()


# In[257]:


X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)


# In[258]:


X_train.isnull().sum()


# In[259]:


X_test.isnull().sum()


# ## Handling Imbalance

# In[260]:


# Copy Splitting data to add model trials
X1 = X.copy()
y1 = y.copy()


# In[261]:


# to test 70:30 data splitting
X1_train1, X1_test1, y1_train1, y1_test1 = train_test_split(X1,y1,test_size=0.3, random_state=42)


# In[262]:


print("X_train: ",X_train.shape)
print("X1_train1: ",X1_train1.shape)
print("X_test: ",X_test.shape)
print("X1_test1: ",X1_test1.shape)


# y_train has a data imbalance problem, where the value of 1 is a minority. This imbalance effect can cause the f1 score to decrease.

# In[263]:


y_train.value_counts(normalize=True) * 100


# In[264]:


X_train[['issue_d','last_pymnt_d','last_credit_pull_d','earliest_cr_line']].info()


# In[265]:


X_train['issue_d:year'] = X_train['issue_d'].dt.year
X_train['issue_d:month'] = X_train['issue_d'].dt.month
X_train.drop(columns='issue_d', inplace=True)

X_train['last_pymnt_d:year'] = X_train['last_pymnt_d'].dt.year
X_train['last_pymnt_d:month'] = X_train['last_pymnt_d'].dt.month
X_train.drop(columns='last_pymnt_d', inplace=True)

X_train['last_credit_pull_d:year'] = X_train['last_credit_pull_d'].dt.year
X_train['last_credit_pull_d:month'] = X_train['last_credit_pull_d'].dt.month
X_train.drop(columns='last_credit_pull_d', inplace=True)

X_train['earliest_cr_line:year'] = X_train['earliest_cr_line'].dt.year
X_train['earliest_cr_line:month'] = X_train['earliest_cr_line'].dt.month
X_train.drop(columns='earliest_cr_line', inplace=True)

X_train.head()


# In[266]:


X_test['issue_d:year'] = X_test['issue_d'].dt.year
X_test['issue_d:month'] = X_test['issue_d'].dt.month
X_test.drop(columns='issue_d', inplace=True)

X_test['last_pymnt_d:year'] = X_test['last_pymnt_d'].dt.year
X_test['last_pymnt_d:month'] = X_test['last_pymnt_d'].dt.month
X_test.drop(columns='last_pymnt_d', inplace=True)

X_test['last_credit_pull_d:year'] = X_test['last_credit_pull_d'].dt.year
X_test['last_credit_pull_d:month'] = X_test['last_credit_pull_d'].dt.month
X_test.drop(columns='last_credit_pull_d', inplace=True)

X_test['earliest_cr_line:year'] = X_test['earliest_cr_line'].dt.year
X_test['earliest_cr_line:month'] = X_test['earliest_cr_line'].dt.month
X_test.drop(columns='earliest_cr_line', inplace=True)

X_test.head()


# In[267]:


# X1_train1['issue_d:year'] = X1_train1['issue_d'].dt.year
# X1_train1['issue_d:month'] = X1_train1['issue_d'].dt.month
# X1_train1.drop(columns='issue_d', inplace=True)

# X1_train1['last_pymnt_d:year'] = X1_train1['last_pymnt_d'].dt.year
# X1_train1['last_pymnt_d:month'] = X1_train1['last_pymnt_d'].dt.month
# X1_train1.drop(columns='last_pymnt_d', inplace=True)

# X1_train1['last_credit_pull_d:year'] = X1_train1['last_credit_pull_d'].dt.year
# X1_train1['last_credit_pull_d:month'] = X1_train1['last_credit_pull_d'].dt.month
# X1_train1.drop(columns='last_credit_pull_d', inplace=True)

# X1_train1['earliest_cr_line:year'] = X1_train1['earliest_cr_line'].dt.year
# X1_train1['earliest_cr_line:month'] = X1_train1['earliest_cr_line'].dt.month
# X1_train1.drop(columns='earliest_cr_line', inplace=True)

# X1_train1.head()


# In[268]:


# X1_test1['issue_d:year'] = X1_test1['issue_d'].dt.year
# X1_test1['issue_d:month'] = X1_test1['issue_d'].dt.month
# X1_test1.drop(columns='issue_d', inplace=True)

# X1_test1['last_pymnt_d:year'] = X1_test1['last_pymnt_d'].dt.year
# X1_test1['last_pymnt_d:month'] = X1_test1['last_pymnt_d'].dt.month
# X1_test1.drop(columns='last_pymnt_d', inplace=True)

# X1_test1['last_credit_pull_d:year'] = X1_test1['last_credit_pull_d'].dt.year
# X1_test1['last_credit_pull_d:month'] = X1_test1['last_credit_pull_d'].dt.month
# X1_test1.drop(columns='last_credit_pull_d', inplace=True)

# X1_test1['earliest_cr_line:year'] = X1_test1['earliest_cr_line'].dt.year
# X1_test1['earliest_cr_line:month'] = X1_test1['earliest_cr_line'].dt.month
# X1_test1.drop(columns='earliest_cr_line', inplace=True)

# X1_test1.head()


# In[269]:


X_train.shape


# In[270]:


y_train.shape


# In[271]:


y_train = y_train.drop(1)
y_train.shape


# In[272]:


# Mengganti semua nilai NaN dengan 0
X_train = X_train.fillna(0)

# Memeriksa apakah ada nilai None dalam dataset
if X_train.isnull().values.any():
    print("Ada nilai None dalam dataset.")


# ### Handling Imbalance Data 80:20

# In[273]:


from imblearn.combine import SMOTETomek

# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_res, y_res = smk.fit_resample(X_train, y_train)


# In[274]:


print("X_resampled.shape:", X_res.shape)
print("y_resampled.shape:", y_res.shape)


# ### Handling Imbalance Data 70:30

# In[275]:


# from imblearn.combine import SMOTETomek

# # Implementing Oversampling for Handling Imbalanced 
# smk = SMOTETomek(random_state=42)
# X1_res1, y1_res1 = smk.fit_resample(X1_train1, y1_train1)

# print("X_resampled.shape:", X1_res1.shape)
# print("y_resampled.shape:", y1_res1.shape)


# ## Modelling

# ### Default Parameter 80:20 balance dataset

# #### Decision Tree

# In[280]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_res,y_res)

y_train_pred_dt = dt.predict(X_res)
y_pred_dt = dt.predict(X_test)


# In[281]:


# score without hyperparameter tuning
dt.score(X_test, y_test)


# #### Random Forest

# In[282]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)

y_train_pred_rf = rf.predict(X_res)
y_pred_rf = rf.predict(X_test)


# In[279]:


# score without hyperparameter tuning
rf.score(X_test, y_test)


# In[284]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluation matrix for Decision tree
print('Accuracy: ', accuracy_score(y_test, y_pred_dt))
print('AUC Score: ', roc_auc_score(y_test, y_pred_dt))
print('f1 Score: ', f1_score(y_test, y_pred_dt))
print('Precission Score: ', precision_score(y_test, y_pred_dt))
print('Recall Score: ', recall_score(y_test, y_pred_dt))
print('---'*10, '\n')

# Evaluation matrix for Random Forest
print('Accuracy: ', accuracy_score(y_test, y_pred_rf))
print('AUC Score: ', roc_auc_score(y_test, y_pred_rf))
print('f1 Score: ', f1_score(y_test, y_pred_rf))
print('Precission Score: ', precision_score(y_test, y_pred_rf))
print('Recall Score: ', recall_score(y_test, y_pred_rf))
print('---'*10, '\n')


# ##### Confusion Matrix

# In[289]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix Decision Tree
cm_test = confusion_matrix(y_test, y_pred_dt)
cm_disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=dt.classes_)
cm_disp_test.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix Decision Tree')
plt.show()

# Confusion Matrix Decision Tree
cm_test_rf = confusion_matrix(y_test, y_pred_rf)
cm_disp_test_rf = ConfusionMatrixDisplay(confusion_matrix=cm_test_rf, display_labels=dt.classes_)
cm_disp_test_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix Random Forest')
plt.show()


# ### Hyperparameter 80:20 balance dataset

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of features to consider at every split
max_features = ['sqrt', 'log2']  # Ganti 'auto' dengan 'sqrt' atau 'log2'

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

