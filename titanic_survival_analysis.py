import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import seaborn as sns
import time
import re
from collections import Counter
'''
pd.options.mode.chained_assignment = None #to hide warnings
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
'''



def exploratory_analysis(df):
	'''This function will provide information about the dataframe to be analyze'''
	print(len(df)) #number of records
	print(df.shape)#number of rows and columns
	print(df.head(5)) #print the first 5 rows
	print(df.columns.values) #get list of column name
	print(df.info()) #index, datatype and memory information
	print(df.describe()) #summary statisitcs for numerical columns
	print(df.apply(pd.Series.value_counts)) #unique values and counts for all columns
	print(df.count())#gives counts of each column...can be used for missing data
	print(df['Age'].min()) #min of a column
	print(df['Survived'].value_counts()) #prints number of rows for a given value
	
	#renaming columns
	#df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
	# Or rename the existing DataFrame (rather than creating a copy) 
	#df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
	
def visualization(df):
	'''This function will do the basic graphing and plotting'''
	f, ax = plt.subplots(3,4,figsize = (20,16)) #subplot divides the grid
	#https://seaborn.pydata.org/generated/seaborn.countplot.html
	sns.countplot('Pclass', data = df, ax = ax[0,0])
	sns.countplot(y = 'Sex', data = df, ax = ax[0,1]) #setting the y attribute displays the graph horizontally
	sns.boxplot(x = 'Pclass', y = 'Age', data = df, ax = ax[0,2])
	sns.countplot('SibSp',hue='Survived',data = df,ax=ax[0,3],palette='husl')
	sns.distplot(df['Fare'].dropna(), ax = ax[2,0], kde=False,color='b')
	sns.countplot('Embarked', data = df, ax = ax[2,2])
	
	sns.countplot('Pclass',hue='Survived',data=df,ax=ax[1,0],palette='husl')
	sns.countplot('Sex',hue='Survived',data=df,ax=ax[1,1],palette='husl')
	sns.distplot(df[df['Survived'] == 0]['Age'].dropna(), ax = ax[1,2], kde = False, color = 'r', bins = 5)
	sns.distplot(df[df['Survived'] == 1]['Age'].dropna(), ax = ax[1,2], kde = False, color = 'g', bins = 5)
	sns.countplot('Parch', hue = 'Survived', data = df, ax = ax[1,3], palette = 'husl')
	sns.swarmplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = df, palette='husl', ax=ax[2,1])
	sns.countplot('Embarked', hue = 'Survived', data = df, ax = ax[2,3], palette = 'husl')
	
	ax[0,0].set_title('Total Passengers by Class')
	ax[0,1].set_title('Total Passengers by Gender')
	ax[0,2].set_title('Age Box Plot By Class')
	ax[0,3].set_title('Survival Rate by SibSp')
	ax[1,0].set_title('Survival Rate by Class')
	ax[1,1].set_title('Survival Rate by Gender')
	ax[1,2].set_title('Survival Rate by Age')
	ax[1,3].set_title('Survival Rate by Parch')
	ax[2,0].set_title('Fare Distribution')
	ax[2,1].set_title('Survival Rate by Fare and Pclass')
	ax[2,2].set_title('Total Passengers by Embarked')
	ax[2,3].set_title('Survival Rate by Embarked')
		
	plt.show()


def detect_outliers(df,n,features):
		outlier_indices = []
		print("inside out")
		#iterate over columns
		for col in features:
			Q1 = np.percentile(df[col],25)
			Q3 = np.percentile(df[col],75)
			IQR = Q3 - Q1
			outlier_step = 1.5*IQR
			#Determine a list of indices of outliers for feature col
			outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
			#append the found outlier indices for col to the list of outlier indices
			outlier_indices.extend(outlier_list_col)
			#select observations containing more than 2 outliers
		outlier_indices = Counter(outlier_indices)
		multiple_outliers = list (k for k, v in outlier_indices.items() if v > n)
		return multiple_outliers


		
def bivariate_statistical_analysis(df):
	sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df)
	
	df['Name_Length'] = train['Name'].apply(len)
	sum_Name = train[["Name_Length","Survived"]].groupby(['Name_Length'], as_index=False).sum()
	average_Name = train[["Name_Length","Survived"]].groupby(['Name_Length'], as_index=False).mean()
	
	fig, (axis1,axis2,axis3) = plt.subplots(3,1,figsize=(18,6))
	sns.barplot(x='Name_Length', y='Survived', data=sum_Name, ax=axis1)
	sns.barplot(x='Name_Length', y='Survived', data=average_Name, ax=axis2)
	sns.pointplot(x='Name_Length', y='Survived', data=train, ax=axis3)
	
	df.loc[df['Name_Length'] <= 23, 'Name_Length'] = 0
	df.loc[(df['Name_Length'] > 23) & (df['Name_Length'] <= 28), 'Name_Length'] = 1
	df.loc[(df['Name_Length'] > 28) & (df['Name_Length'] <= 40), 'Name_Length'] = 2
	df.loc[df['Name_Length'] > 40, 'Name_Length'] = 3
	#print(df['Name_Length'].value_counts())
	
	#gender (Sex)
	df['Sex'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
	#df.replace(['female','male'], [0,1])
	
	#age
	a = sns.FacetGrid(df, hue="Survived", aspect=6)
	a.map(sns.kdeplot, 'Age', shade=True)
	a.set(xlim=(0, df['Age'].max()))
	a.add_legend()
	
	age_avg = df['Age'].mean()
	age_std = df['Age'].std()
	age_null_count = df['Age'].isnull().sum()
	age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
	df['Age'][np.isnan(df['Age'])] = age_null_random_list
	df['Age'] = df['Age'].astype(int)
	df.loc[df['Age'] <= 14, 'Age'] = 0
	df.loc[(df['Age'] > 14) & (df['Age'] <= 30), 'Age'] = 5
	df.loc[(df['Age'] > 30) & (df['Age'] <= 40), 'Age'] = 1
	df.loc[(df['Age'] > 40) & (df['Age'] <= 50), 'Age'] = 3
	df.loc[(df['Age'] > 50) & (df['Age'] <= 60), 'Age'] = 2
	df.loc[df['Age'] > 60, 'Age'] = 4
	print(df['Age'].value_counts())
	
	
	#Family: SibSp and Parch
	df['Fare'] = df['Fare'] = df['Fare'].fillna(df['Fare'].median())
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsAlone'] = 0
	df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
	
	df['Boys'] = 0
	df.loc[(df['Age'] == 0) & (df['Sex'] == 1), 'Boys'] = 1
	
	fig, (axis1, axis2) = plt.subplots(1,2,figsize=(18,6))
	sns.barplot(x="FamilySize", y="Survived", hue="Sex", data=df, ax=axis1)
	sns.barplot(x="IsAlone", y="Survived", hue="Sex", data=df, ax=axis2)
	
	
	#explore fare distribution
	fig,axis = plt.subplots(1,1,figsize=(18,6))
	g = sns.distplot(df['Fare'], color="m", label="Skewness : %.2f"%(df["Fare"].skew()))
	g = g.legend(loc = "best")
	
	#apply log to fare to reduce skewness distribution
	df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
	a4_dims = (20,6)
	fig, ax = plt.subplots(figsize = a4_dims)
	g = sns.distplot(df['Fare'][df['Survived'] == 0], color='r', label="Skewness : %.2f"%(df['Fare'].skew()), ax=ax	)
	g = sns.distplot(df['Fare'][df['Survived'] == 1], color='b', label="Skewness : %.2f"%(df['Fare'].skew()))
	g = g.legend(["Not Survived", "Survived"])
	
	df.loc[df['Fare'] <= 2.7, 'Fare'] = 0
	df.loc[df['Fare'] > 2.7, 'Fare'] = 3
	
	#count of survivors by Fare
	print(df.loc[df['Fare'] == 0, 'Survived'].sum())
	print(df.loc[df['Fare'] == 3, 'Survived'].sum())
	
	#Cabin
	df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
	#df[['Has_Cabin','Survived']].groupby('Has_Cabin'), as_index=False).sum().sort_values(by='Survived', ascending=False)	
	print(df[["Has_Cabin", "Survived"]].groupby(['Has_Cabin'], as_index=False).sum().sort_values(by='Survived', ascending=False))
	
	#Embarked
	df['Embarked'] = df['Embarked'].fillna('S')
	#mapping embarked
	df['Embarked'] = df['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2}).astype(int)
	df_pivot = pd.pivot_table(df, values = 'Survived', index = ['Embarked'], columns='Pclass', aggfunc=np.mean, margins=True)
	print(df_pivot)
	
	df['Embarked'] = df['Embarked'].replace(['0','2'], '0')
	df['Fare'].value_counts()

#function to extract titles from passenger names
def get_title(name):
	title_search = re.search('([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

def group_titles(df):
	df['Title'] = df['Title'].replace(['Mrs', 'Miss'], 'MM')
	df['Title'] = df['Title'].replace(['Dr', 'Major', 'Col'], 'DMC')
	df['Title'] = df['Title'].replace(['Don', 'Rev', 'Capt', 'Jonkheer'],'DRCJ')
	df['Title'] = df['Title'].replace(['Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'Countess'],'MMLSMC' )
	# Mapping titles
	title_mapping = {"MM": 1, "Master":2, "Mr": 5, "DMC": 4, "DRCJ": 3, "MMLSMC": 0}
	df['Title'] = df['Title'].map(title_mapping)
	df['Title'] = df['Title'].fillna(3)


def extractdeck(df):
	deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
	df['Cabin'] = df['Cabin'].fillna("U0")
	df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
	df['Deck'] = df['Deck'].map(deck)
	df['Deck'] = df['Deck'].fillna(0)
	df['Deck'] = df['Deck'].astype(int)
	

def descriptive_stats(train):
	print(train.describe())
	print(train[train['Pclass'] == 1]['Survived'].sum()/len(train[train['Pclass'] == 3]))
	#df[df['Survived'] == 0]['Age']
	fig, (axis1,axis2) = plt.subplots(1,2,figsize=(18,6))
	sns.barplot(x="Embarked",y="Survived", hue="Sex", data=train, ax=axis1)
	sns.barplot(x="Age",y="Survived", hue="Sex", data=train, ax=axis1)
	
	train['boys'] = 0
	train.loc[(train['Age'] == 0) & (train['Sex'] == 1), 'Boys'] = 1
	
	
	print(train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	
	
	east lot - 80
	
	spring lot - 65
	

	
def pearson_correlation(train):
	colormap = plt.cm.RdBu
	plt.figure(figsize=(14,12))
	plt.title('Pearson correlation of Features', y=1.05, size=15)
	sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
	
	g = sns.pairplot(train, hue='Survived', palette='husl', diag_kind = 'kde')
	g.set(xticklabels=[])

train = pd.read_csv('C:/Users/jainv6/Documents/Learning/python/DataScience/Titanic/train.csv')
test = pd.read_csv('C:/Users/jainv6/Documents/Learning/python/DataScience/Titanic/test.csv')
full_data = [train,test]

#exploratory_analysis(train)
#visualization(train)

#detect outliers from Age, SibSp, Parch and Fare
#outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
#print(train.loc[outliers_to_drop]) #show the outlier rows

#drop outliers..to drop columns - df.drop(['B', 'C'], axis=1), to drop rows - df.drop([0, 1])
#train = train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)

#https://seaborn.pydata.org/tutorial/distributions.html

#The approach to to complete missing data is to impute using mean, median, or mean + randomized standard deviation.
# the fillna function: dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

bivariate_statistical_analysis(train)
train['Title'] = train['Name'].apply(get_title)

fig, (axis1) = plt.subplots(1, figsize=(18,6))
sns.barplot(x="Title", y="Survived", data=train, ax=axis1)
group_titles(train)
extractdeck(train)

descriptive_stats(train)
exploratory_analysis(train)
#plt.show()

#dropping unnecessary features
print(train.columns.values)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Boys', 'IsAlone', 'Embarked']

train = train.drop(drop_elements, axis = 1)

pearson_correlation(train)
plt.show()