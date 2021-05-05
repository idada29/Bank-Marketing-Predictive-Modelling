# Bank-Marketing-Predictive-Modelling
Exploring classification algorithms for prediction of term deposit subscription

Section 1 : Importing packages and modules
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib as mpl
import scipy

#This is to label all categorical data
from sklearn.preprocessing import LabelEncoder 

#This is plotly module used for plotting
import plotly.express as px

import IPython
import sklearn

# Importing the required packages for Decision Tree
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

#This packages must be installed for some visualizations to work properly
#!pip install cufflinks
#!pip install chart_studio
#!pip install plotly

#This is needed to make use of pandas profilling to show the break down of the data
#conda install -c conda-forge pandas-profiling

# This is need to print out notebook as pdf
#pip install -U notebook-as-pdf
#conda install nbconvert

# Display the Dive visualization for the training data.
from IPython.core.display import display, HTML
#!pip install facets-overview



# The two csv found on the link were combined to have an increased observation
banking_data = pd.read_csv("C:\\Users\\dadai\\Downloads\\Machine Learning\\ICA\\BANKING DATA FROM UCI\\BANK MARKETING\\bank-combined.csv")

banking_data.shape

from pandas_profiling import ProfileReport
banking_data.profile_report() 

# This piece of code was used to revalidate that this items here represented 
# as duplicated above doesn't mean they are actually duplicated rows entirely 
#but we have some rows whose values tallies with some values from other 
#rows for more than 5-7 columns

duplicateRowsDF = banking_data[banking_data.duplicated()]
duplicateRowsDF.head(10)

banking_data.describe(include= "all")

banking_data.info()

# Check for missing values in the training dataset, 
# output indicates that our dataset has no null values

print(banking_data.isnull().values.any())
print(" ")
print(banking_data.isnull().sum())

# EXPLORATORY DATA ANALYSIS (EDA) AND DATA PREPROCESSING

distinct_marital = banking_data.marital.value_counts()
print('The distinct value of marital status \n{}'.format(distinct_marital))
print('')

distinct_education = banking_data.education.value_counts()
print('Diiferent education types within the data: \n{}'.format(distinct_education))
print('')

distinct_default = banking_data.default.value_counts()
print('Any previous defaulted obligations: \n{}'.format(distinct_default))
print('')

distinct_housing = banking_data.housing.value_counts()
print('Owners of houses: \n{}'.format(distinct_housing))
print('')

distinct_loans = banking_data.loan.value_counts()
print('Have outstanding loans: \n{}'.format(distinct_housing))
print('')

distinct_contact = banking_data.contact.value_counts()
print('The mode of contact and frequency: \n{}'.format(distinct_contact))
print('')

distinct_outcome = banking_data.poutcome.value_counts()
print('Outcome for previous marketing campaigns: \n{}'.format(distinct_outcome))
print('')


# I noticed an imbalanced distribution of class of our target feature. I will be attending to this few steps below 
target_ratio = banking_data['y'].value_counts()
print("The classes of our binary target: \n{}".format(target_ratio))


# The dataset target is really skewed and class of target is imbalance as shown below, 
# however using KNN to attend to this by selecting samples within clusters formed
# in the majority class

fig = px.pie(banking_data['y'].value_counts().reset_index(), values = 'y',
             names = ['no', 'yes'])
fig.update_traces(textposition = 'inside', textinfo = 'percent + label', hole = 0.6, 
                  marker = dict(colors = ['#0e4bef','#ffcccb'],
                                line = dict(color = 'white', width = 3)))
# Add text labelling
fig.update_layout(annotations = [dict(text = 'Target', 
        x = 0.5, y = 0.5,font_size = 24, showarrow = False, 
        font_family = 'Times New Roman',font_color = 'black')],showlegend = False)           
fig.show()

# calculate correlation matrix
grid_kws = {'height_ratios':(0.9,0.05),'hspace':0.3}
f,(ax,cbar_ax)=plt.subplots(2,gridspec_kw=grid_kws)

corr = banking_data.corr()

ax= sns.heatmap(corr,annot=True,annot_kws={'fontsize':12},
                linewidths= 1,ax=ax,cbar_ax=cbar_ax,
                cbar_kws={'orientation':'horizontal'})

corr = banking_data.corr()
mask= np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    f,ax= plt.subplots(figsize=(12,5))
    ax= sns.heatmap(corr,mask=mask,annot=True,annot_kws={'fontsize':10},linewidths= 1,vmax=7,square = True)

# USING CLUSTER & SAMPLING TO ADDRESS IMBALANCED CLASS WITHIN THE DATASET

feature_names = banking_data.columns

# In order to further understand our dataset, identify the columns with numeric data against categorical to enable proper grouping
# and exploration.

numericColumns= banking_data._get_numeric_data().columns
nCols = list(numericColumns[0:])
categoricalColumns = list(set(feature_names) - set(numericColumns))
categoricalColumns.remove('y')

print ('The numericColumns are: {}'.format(nCols))
print('\n')
print('The categoricalColumns are: {}'.format(categoricalColumns))

# We need to convert the categorical data into indicator variables (0 or 1) and also scale the numerical features
categorical_features = pd.get_dummies(banking_data[categoricalColumns])


# What the above those is that: it selects on the cateogrical columns, then creates a separate column for each of the 
# elements/values within that column assisgning a value of 1 to it if present and zeros (0) to others. Thus increasing the number 
# of columns but having all the category now represented as 0 or 1

categorical_features.head()

# Next is to scale numeric features, for this I made use of standardScaler()
from sklearn.preprocessing import StandardScaler
numeric_features = banking_data[numericColumns]
scaler = StandardScaler()

#The scaled features is then converted back to dataFrame to enable me merge/concatenate it with the categorical features
scaled_numeric_features = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns)

# Concatenate the the categorical_feature and scaled numeric features
new_data = pd.concat([categorical_features, scaled_numeric_features], axis=1)
new_data.shape

# You will observe at this point that the number of columns increased, hence a need for PCA to determine component selection / subset
# which best explains the data

new_data.head()

# Feature Extraction with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca_transformed_data = pca.fit_transform(new_data)

df = pd.DataFrame(pca_transformed_data)
headers = ['PC1','PC2','PC3','PC4']
df.columns = headers


df.head()

#Extract all values from the 3 principal components
PCA_values = df[['PC1','PC2','PC3']].values

# Import KElbowVisualizer and KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Use yellowbrick cluster to determine the number of cluster based on nearest neighbours
visualizer = KElbowVisualizer(KMeans(),k=(2,11))
visualizer.fit(PCA_values)
visualizer.show(outpath='elbowplot.png')

n_clusters = 4 #From the elbow plot

from scipy.cluster.vq import kmeans2
#To find out the centroid which would enable us fit properly with KMeans
centroid, label = kmeans2(pca_transformed_data,n_clusters, minit='points')
centroid


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, max_iter=10000, verbose=1, n_jobs=4, init=centroid)

clusters = kmeans.fit_predict(df)

#Add a new columns for the predicted KMeans cluster prediction
df['cluster'] = clusters
df.info()
df.head()


# Plot to visualize display the relationships of the principal components 

fig, ax = plt.subplots(figsize = (5, 5))
plt.scatter(df[['PC2']], df[['PC3']], c=df['cluster'])
ax.set_xlabel("PC2",fontsize=12)
ax.set_ylabel("PC3",fontsize =12)
ax.set_title("Comparing Principal components",fontsize = 14)


fig, ax = plt.subplots(figsize = (5, 5))
plt.scatter(df['PC1'], df['PC2'], c=df['cluster'])
ax.set_xlabel("PC1",fontsize=12)
ax.set_ylabel("PC3",fontsize =12)
ax.set_title("Comparing Principal components PC1 and PC2",fontsize = 14)

fig, ax = plt.subplots(figsize = (5, 5))
plt.scatter(df['PC1'], df['PC2'], c=df['cluster'])
ax.set_xlabel("PC1",fontsize=12)
ax.set_ylabel("PC2",fontsize =12)
ax.set_title("Comparing Principal components PC1 and PC2",fontsize = 14)

# Convert target no and yes into 0 and 1 and store in a variable to be used later
target_labels = banking_data['y'].map({'no':0, 'yes':1})

# Select a sample from each quarter of the minor class
minority_class = target_labels.value_counts()[1]

# Divide the minority class and discard the remainder or decimal 
sample_size = minority_class//n_clusters

print("The sample size to be considered when undersampling the majority class is: {}".format(sample_size))


#We need to select random data points samples of the majority class 0 or no
my_list = []

for value in range(0,n_clusters):
    # create an array of index for which selects majority target_labels = 0 and concides 
    # with iteration of value of the different clusters
    majority_class = df[(target_labels==0) & (df['cluster'] == value)].index
    
    # Using np.random.choice function, to select a random size/samples which is the 
    # same as the size of samples selected from the minority_samples 
    my_list.append(np.random.choice(majority_class, sample_size))   
        
# Finally, using np.ravel, the shape list stored in my_list is reshaped into a 
# flattened aray without any specific order      
my_list = np.ravel(my_list)

# The items in the array within this represents only 1452 samples of each clusters of the majority class 'no' 
print(np.count_nonzero(my_list))

# The resampled data is the concatenation of all the minority class and the unsampled majority class (no)
resampled_data = pd.concat([banking_data.iloc[my_list], banking_data[banking_data['y'] == 'yes']])
print(resampled_data['y'].value_counts())

# A FEW THINGS TO NOTE:

# 1. Resampled_data would always be a new subset of the previous clusters because we induced np.random in randomly selecting the different values of variable choices and new samples above
# 2. In order to have the record of the resampled_data you initated at the point, we would be saving the resampled_data to your local path to enable us reaccess whenever we need it.


resampled_data.to_csv("Resampled_data.csv")

# In order the maintain the same results during test and after test, the file used and saved locally would be read in again
# to enable us work with the same file


#resampled_data = pd.read_csv("C:\\Users\\dadai\\Documents\\python_bootcamp\\Resampled_data.csv")


resampled_data.head()

# The final results of target class distribution is shown below

fig = px.pie(resampled_data['y'].value_counts().reset_index(), values = 'y',
             names = ['no', 'yes'])
fig.update_traces(textposition = 'inside', 
                  textinfo = 'percent + label', 
                  hole = 0.6, 
                  marker = dict(colors = ['#0e4bef','#ffcccb'],
                                line = dict(color = 'white', width = 3)))

# Add text labelling
fig.update_layout(annotations = [dict(text = 'Target', 
                                      x = 0.5, y = 0.5,
                                      font_size = 24, showarrow = False, 
                                      font_family = 'Times New Roman',
                                      font_color = 'black')],
                  showlegend = False)
                  
fig.show()

# SOME MORE DATA EXPLORATION AND VISUALIZATION OF RESAMPLED_DATA TO BE USED FOR TRAINING AND TESTING

# A plot to show the relationship of different features as compared to the target
sns.set(style="ticks")
sns.pairplot(resampled_data, hue="y", palette="Set1")

# Visualizing relationship between categorical columns and balance
fig = px.box(resampled_data,x = "job",color="y",orientation='v',title="Distribution of target outcome by different job types")
fig.show()

# We have to also show Graph to show the frequency of target outcome by different job types below:
graphs = pd.concat([resampled_data['job'][resampled_data.y=='yes'].value_counts(),resampled_data['job'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome responses by different job types',ylabel= 'Count',figsize = (15, 5))


# From the output we see that more single and divorced tends to want and buy in to a marketed products


#Comparing account balances for customers with house loans or not as against our target outcome
fig = px.box(resampled_data,x = "housing",y="balance",color="y",points = "all",orientation='v',title="Comparing account balances by customers with house loans or not as against our target outcome")
fig.show()

# We have to also show Graph to show the frequency of target outcome by different housing :
graphs = pd.concat([resampled_data['housing'][resampled_data.y=='yes'].value_counts(),resampled_data['housing'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome responses by if customers currently has a house loan or not',ylabel= 'Count',figsize = (10, 5))



#Distribution of personal loans and balance compared to the target outcome
fig = px.box(resampled_data,x = "loan",y="balance",color="y",points = "all",orientation='v',title="Distribution of personal loans and balance with respect to the target outcome")
fig.show()

# We have to also show Graph to show the frequency of target outcome by different customers having a pending loan or not:
graphs = pd.concat([resampled_data['loan'][resampled_data.y=='yes'].value_counts(),resampled_data['loan'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome responses by if customers is a beneficiary of a personal loan with the bank or not',ylabel= 'Count',figsize=(10,5))



#Comparing account balances by marital status as against our target outcome
fig = px.box(resampled_data,x = "marital", y = "balance",color="y",points="all",title="Comparing account balances by marital status as against our target outcome")
fig.show()

# We have to also show Graph to show the frequency of target outcome by different marital status:
graphs = pd.concat([resampled_data['marital'][resampled_data.y=='yes'].value_counts(),resampled_data['marital'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome count by different marital status',ylabel= 'Count',figsize=(10,5))


# From the output we see that more single and divorced tends to want and buy in to a marketed products

# Comparing account balances by education type
fig= px.box(resampled_data,x = "education", y = "balance",color='y',title="Comparing account balances by education type",notched=True,points = "all")
fig.show()

# The graph above shows the distribution of responses by different education institution, however the bar chart below drills down
# to comparing by count of the target outcome.

graphs = pd.concat([resampled_data['education'][resampled_data.y=='yes'].value_counts(),resampled_data['education'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome count by different educational institution',ylabel= 'Count',figsize=(10,5))

# Comparing account balances by contact type

fig = px.box(resampled_data,x = "contact", y = "balance",color='y',points= "all",notched=True,orientation='v',title="Comparing account balances by contact type")
fig.show()

# We have to also show Graph to show the frequency of target outcome by different contact type below:
graphs = pd.concat([resampled_data['contact'][resampled_data.y=='yes'].value_counts(),resampled_data['contact'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs.plot(kind='bar',color=['red','blue'],title= 'Target outcome count by different contact type',ylabel= 'Count',figsize=(10,5))


import plotly.offline as py
from plotly.offline import iplot
import plotly.graph_objs as go

#This plot compares balance and previous contact compared with the target outcomes
resampled_data_pr_1 = resampled_data[resampled_data.y == 'yes']
resampled_data_pr_2 = resampled_data[resampled_data.y == 'no']

# For target outcome 'yes'
trace1 =go.Scatter(
                    y = resampled_data_pr_1.balance,
                    x = resampled_data_pr_1.previous,
                    mode = "markers",
                    name = "Target: yes",
                    marker = dict(color = 'rgba(240, 136, 200, 0.8)'),
                    text= resampled_data_pr_1.y)
# For target outcome 'no'
trace2 =go.Scatter(
                    y = resampled_data_pr_2.balance,
                    x = resampled_data_pr_2.previous,
                    mode = "markers",
                    name = "Target: no",
                    marker = dict(color = 'rgba(0, 130, 200, 0.8)'),
                    text= resampled_data_pr_2.y)

data = [trace1, trace2]

# Plotting the layout
layout = dict(title = 'Balance - previous contact - target',
              xaxis= dict(title= 'Previous contact',
                          ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Balance',
                          ticklen= 5,zeroline= False),
             autosize=False,
             width=700,
             height=450,)
fig = dict(data = data, layout = layout)
    
iplot(fig)

#This plot compares balance and age compared with the target outcomes
resampled_data_pr_1 = resampled_data[resampled_data.y == 'yes']
resampled_data_pr_2 = resampled_data[resampled_data.y == 'no']

# For target outcome 'yes'
trace1 =go.Scatter(
                    y = resampled_data_pr_1.balance,
                    x = resampled_data_pr_1.age,
                    mode = "markers",
                    name = "Target: yes",
                    marker = dict(color = 'rgba(240, 136, 200, 0.8)'),
                    text= resampled_data_pr_1.y)
# For target outcome 'no'
trace2 =go.Scatter(
                    y = resampled_data_pr_2.balance,
                    x = resampled_data_pr_2.age,
                    mode = "markers",
                    name = "Target: no",
                    marker = dict(color = 'rgba(0, 130, 200, 0.8)'),
                    text= resampled_data_pr_2.y)

data = [trace1, trace2]

# Plotting the layout
layout = dict(title = 'Balance - Age - target',
              xaxis= dict(title= 'Age',
                          ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Balance',
                          ticklen= 5,zeroline= False),
             autosize=False,
             width=700,
             height=450,)
fig = dict(data = data, layout = layout)
    
iplot(fig)

# Looking at the numerical features and its diverse spread
plt.style.context('dark_background')
resampled_data.hist(bins=20, figsize=(14,10), color='blue')


fig = px.violin(resampled_data, y="y", x="balance", box=True, points="all",hover_data=resampled_data.columns,color="y")
fig.show()


#To check balances distribution by months,we use a go.Violin 

import plotly.graph_objects as go

fig = go.Figure()

months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

for value in months:
    fig.add_trace(go.Violin(x=resampled_data['month'][resampled_data['month'] == value],
                            y=resampled_data['balance'][resampled_data['month'] == value],name=value,
                            box_visible=True,
                            meanline_visible=True))

fig.show()

graphs = pd.concat([resampled_data['month'][resampled_data.y=='yes'].value_counts(),resampled_data['month'][resampled_data.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs = pd.DataFrame(graphs)
graphs.reset_index(inplace=True)
headers = ['month','yes','no']
graphs.columns = headers

# multiple line plots
plt.figure(figsize=(15, 7))
plt.plot('month', 'yes', data=graphs, marker='o', color='red', linewidth=2)
plt.plot('month', 'no', data=graphs, marker='x', color='blue', linewidth=2, linestyle='dashed', label="no")
# show legend
plt.legend()
# show graph
plt.show()


# Selecting all data for Month = Oct since I observed from line that represents all the monthly outcomes, that 'October' had the
# highest 'Yes' and least 'No'. The chart below is to explore data specific for month 'Ocotber'

# Select all columns for October
Oct = resampled_data.loc[resampled_data['month'] == 'oct']
Oct.head()

#Concatenating a new dataframe 'graph' for all counts of target by days in the variable 'Oct' and give the dataframe a right header
graphs = pd.concat([Oct['day'][Oct.y=='yes'].value_counts(),Oct['day'][Oct.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs = pd.DataFrame(graphs)
graphs.reset_index(inplace=True)
headers = ['days','yes','no']
graphs.columns = headers

# Replace all values 'NaN' with 'Zero' which indicates that for that day no value was recorded for that outcome.
graphs.replace(np.nan,0,inplace = True)
graphs.head()


# multiple line plots. I observed that we tend to have more postive responses towards the end of the month from 21st-31st
# Also, due to the pandas version 1.0.5 used for this work, the plot below cannot accept xlabel = 'counts of responses' and 
# ylabel = 'days within the month' as part parameters for plotting. Hence the lack of labels

plt.figure(figsize=(15, 7))
plt.plot('days', 'yes', data=graphs, marker='o', color='red', linewidth=2)
plt.plot('days', 'no', data=graphs, marker='x', color='blue', linewidth=2, linestyle='dashed', label="no")
plt.xticks(np.arange(0, len(graphs['days'])+5, 1))

# show legend
plt.legend()
# show graph
plt.show()

# To show responses within september also, as it has a similarly response rate with October. Observations within days in Sep
# includes a higher positive response pattern and lesser 'no' compared to October

# Select all columns for October
Sep = resampled_data.loc[resampled_data['month'] == 'sep']
Sep.head()

#Concatenating a new dataframe 'graph' for all counts of target by days in the variable 'Sep' and give the dataframe a right header
graphs = pd.concat([Sep['day'][Sep.y=='yes'].value_counts(),Sep['day'][Sep.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs = pd.DataFrame(graphs)
graphs.reset_index(inplace=True)
headers = ['days','yes','no']
graphs.columns = headers

# Replace all values 'NaN' with 'Zero' which indicates that for that day no value was recorded for that outcome.
graphs.replace(np.nan,0,inplace = True)
graphs.head()


plt.figure(figsize=(15, 7))
plt.plot('days', 'yes', data=graphs, marker='o', color='red', linewidth=2)
plt.plot('days', 'no', data=graphs, marker='x', color='blue', linewidth=2, linestyle='dashed', label="no")
plt.xticks(np.arange(0, len(graphs['days'])+5, 1))

# show legend
plt.legend()
# show graph
plt.show()

# May has a very high degree of "YES" but also has a high degree of "NO"

# Select all columns for May
May = resampled_data.loc[resampled_data['month'] == 'may']
May.head()

#Concatenating a new dataframe 'graph' for all counts of target by days in the variable 'May' and give the dataframe a right header
graphs = pd.concat([May['day'][May.y=='yes'].value_counts(),May['day'][May.y=='no'].value_counts()],axis=1)
headers = ['yes','no']
graphs.columns = headers
graphs = pd.DataFrame(graphs)
graphs.reset_index(inplace=True)
headers = ['days','yes','no']
graphs.columns = headers

# Replace all values 'NaN' with 'Zero' which indicates that for that day no value was recorded for that outcome.
graphs.replace(np.nan,0,inplace = True)
graphs.head()


plt.figure(figsize=(15, 7))
plt.plot('days', 'yes', data=graphs, marker='o', color='red', linewidth=2)
plt.plot('days', 'no', data=graphs, marker='x', color='blue', linewidth=2, linestyle='dashed', label="no")
plt.xticks(np.arange(0, len(graphs['days'])+5, 1))

# show legend
plt.legend()
# show graph
plt.show()

# Output from below shows customers with houses loans tends to have more money within the bank than those without houses

fig, ax = plt.subplots(figsize = (10, 5))
resampled_data['balance'][resampled_data.housing=='no'].plot(kind='hist',color='blue')
resampled_data['balance'][resampled_data.housing=='yes'].plot(kind='hist',color='skyblue')
plt.legend(['yes','no'])
plt.xlabel('Balance',size =14)


# I noticed more duration during marketing was spent on people with housing loans

fig, ax = plt.subplots(figsize = (10, 5))
resampled_data['duration'][resampled_data.housing=='no'].plot(kind='hist',color='blue')
resampled_data['duration'][resampled_data.housing=='yes'].plot(kind='hist',color='skyblue')
plt.legend(['yes','no'])
plt.xlabel('Duration',size =14)


# To visualize age distribution by education for those who responded as Yes to the campaign


df = resampled_data.loc[resampled_data["y"] == "yes"]

occupations = resampled_data["education"].unique().tolist()

# Get the ages by education
primary = df["age"].loc[df["education"] == "primary"].values
secondary = df["age"].loc[df["education"] == "secondary"].values
tertiary = df["age"].loc[df["education"] == "tertiary"].values
unknown = df["age"].loc[df["education"] == "unknown"].values

ages = [primary,secondary,tertiary,unknown]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',
          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 
          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',
         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']

newdata_pack = []

for xd, yd, cls in zip(occupations, ages, colors):
        newdata_pack.append(go.Box(y=yd,name=xd,boxpoints='all',jitter=0.5,whiskerwidth=0.2,fillcolor=cls,marker=dict(size=2,),
            line=dict(width=1),))

layout = go.Layout(title='Distribution of Ages by Education',yaxis=dict(autorange=True,showgrid=True,zeroline=True,dtick=5,gridcolor='rgb(255, 255, 255)',gridwidth=1,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2,),
    margin=dict(l=40,r=30,b=80,t=100,),paper_bgcolor='rgb(224,255,246)',plot_bgcolor='rgb(251,251,251)',showlegend=True)

fig = go.Figure(data=newdata_pack, layout=layout)
fig.show()

# To visualize age distribution of marital status for those who responded as YES to the campaign


df = resampled_data.loc[resampled_data["y"] == "yes"]

maritals = resampled_data["marital"].unique().tolist()

# Get the ages by education
single = df["age"].loc[df["marital"] == "single"].values
married = df["age"].loc[df["marital"] == "married"].values
divorced = df["age"].loc[df["marital"] == "divorced"].values


ages = [single,married,divorced]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',
          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 
          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',
         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']

newdata_pack = []

for xd, yd, cls in zip(maritals, ages, colors):
        newdata_pack.append(go.Box(y=yd,name=xd,boxpoints='all',jitter=0.5,whiskerwidth=0.2,fillcolor=cls,marker=dict(size=2,),
            line=dict(width=1),))

layout = go.Layout(title='Distribution of Ages by Marital Status',yaxis=dict(autorange=True,showgrid=True,zeroline=True,dtick=5,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2,
    ),margin=dict(l=40,r=30,b=80,t=100,),paper_bgcolor='rgb(224,255,246)',plot_bgcolor='rgb(251,251,251)',showlegend=True)

fig = go.Figure(data=newdata_pack, layout=layout)
fig.show()

# To visualize age distribution by education of those who responded as YES to the campaign


df = resampled_data.loc[resampled_data["y"] == "yes"]

job_type = resampled_data["job"].unique().tolist()

# Get the ages by job type
management = df["age"].loc[df["job"] == "management"].values
admin = df["age"].loc[df["job"] == "admin."].values
technician = df["age"].loc[df["job"] == "technician"].values
services = df["age"].loc[df["job"] == "services"].values
retired = df["age"].loc[df["job"] == "retired"].values
blue_collar = df["age"].loc[df["job"] == "blue-collar"].values
unemployed = df["age"].loc[df["job"] == "unemployed"].values
entrepreneur = df["age"].loc[df["job"] == "entrepreneur"].values
housemaid = df["age"].loc[df["job"] == "housemaid"].values
self_employed = df["age"].loc[df["job"] == "self-employed"].values
student = df["age"].loc[df["job"] == "student"].values
unknown = df["age"].loc[df["job"] == "unknown"].values


ages = [management,blue_collar,technician,admin,services,retired,self_employed,student,entrepreneur,unemployed,housemaid,unknown]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)',
          'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 
          'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)',
         'rgba(229, 126, 56, 0.5)', 'rgba(229, 56, 56, 0.5)',
         'rgba(174, 229, 56, 0.5)', 'rgba(229, 56, 56, 0.5)']

newdata_pack = []

for xd, yd, cls in zip(job_type, ages, colors):
        newdata_pack.append(go.Box(y=yd,name=xd,boxpoints='all',jitter=0.5,whiskerwidth=0.2,fillcolor=cls,marker=dict(size=2,),
            line=dict(width=1),))

layout = go.Layout(title='Distribution of Ages by Job type',yaxis=dict(autorange=True,showgrid=True,zeroline=True,dtick=5,gridcolor='rgb(255, 255, 255)',gridwidth=1,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2,),
    margin=dict(l=40,r=30,b=80,t=100,),
    paper_bgcolor='rgb(224,255,246)',
    plot_bgcolor='rgb(251,251,251)',
    showlegend=True)

fig = go.Figure(data=newdata_pack, layout=layout)
fig.show()

# FEATURE SELECTION PROCESSING

# Creating label encoders to treat all categorical variables, the reason for selecting label encoder is because they features 
# in most columns are either ordinal or just binary which prevents the possiblity of model wrongly capturing their 
# relationship in an order
labelencoder = LabelEncoder()
resampled_data["job"] = labelencoder.fit_transform(resampled_data["job"])
resampled_data["marital"] = labelencoder.fit_transform(resampled_data["marital"])
resampled_data["education"] = labelencoder.fit_transform(resampled_data["education"])
resampled_data["default"] = labelencoder.fit_transform(resampled_data["default"])
resampled_data["housing"] = labelencoder.fit_transform(resampled_data["housing"])
resampled_data["loan"] = labelencoder.fit_transform(resampled_data["loan"])
resampled_data["contact"] = labelencoder.fit_transform(resampled_data["contact"])
resampled_data["month"] = labelencoder.fit_transform(resampled_data["month"])
resampled_data["poutcome"] = labelencoder.fit_transform(resampled_data["poutcome"])

resampled_data = banking_data = pd.read_csv("C:\\Users\\dadai\\Documents\\python_bootcamp\\resampled_data.csv")
#resampled_data.to_csv('resampled_data.csv')

working_data = resampled_data.copy()

working_data.drop(['pdays'], axis=1,inplace = True)
working_data.drop(['y'], axis=1,inplace = True)


from sklearn.preprocessing import StandardScaler
scaled_data = pd.DataFrame(StandardScaler().fit_transform(working_data),columns=working_data.columns)

jsonstr = resampled_data.to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=jsonstr)
display(HTML(html))

scaled_data.head()

# Feature Extraction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)
pca_data = pca.fit(scaled_data)
PCA_Component = pd.DataFrame(pca_data.components_, columns=list(scaled_data.columns))

# Calculate the variance, however multiply by 100 to make it a percentage, and next step is to label each principal component output in the numpy array 'percentage_variance'
percentage_variance =np.round(pca.explained_variance_ratio_*100, decimals =1)
labels = ['PC'+ str(x) for x in range(1,len(percentage_variance)+1)]

plt.figure(figsize=(12, 5))
plt.bar(x=range(1,len(percentage_variance)+1), height = percentage_variance, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# The scree plot below is the chart above in a line plot to enable me see the elbow curve clearly which indicates the best size of principal components that explained a proportion of variance
PC_values = np.arange(pca.n_components_) + 1

plt.figure(figsize=(12, 5))
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], ["First component", "Second component","Third component","Fourth component","Fifth component","Sixth component","Seventh component","Eighth component",
                                                 "Ninth component","Tenth component","Eleventh component","Twelveth component","Thirteenth component","Fourteenth component","Fifteenth component"])


plt.colorbar()
plt.xticks(range(len(scaled_data.columns)),scaled_data.columns, rotation=90, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")

# summarize components
print("Explained Variance: %s" % pca_data.explained_variance_ratio_)
print(" ")

# The threshold I decided to set is the point/features where the cumulative proportion of the variance explained surpasses 70%
print ("Cumulative Prop. Variance Explained: ", np.cumsum(pca.explained_variance_ratio_))


print (" ")
print(pca_data.components_)

## DECISION TREE CLASSIFER

GridSearch Cross Validation

from sklearn import tree
from sklearn.model_selection import GridSearchCV

# This Gridsearch cv process ran for approximately 2 minutes 45 seconds, however this can be reduced by using RandomizedSearcH CV

# Add a new column 'class' housing the newly relabelled target column for this process
resampled_data['class'] = resampled_data['y'].map({'no':0,'yes':1})

# Select all features excluding column 'y'
x = working_data.values[:,0:15]
target = resampled_data['class']

# The following hyperparameters would be iterated over to select the best possible combination
parameters = {'criterion':("entropy","gini"),'splitter':("random","best"),'max_depth':range(3,20),
              'min_samples_leaf':range(14,20),'random_state': range(90,100)}

# The classifier is created, in this case our model is a decision tree
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4,cv=5)
clf.fit(X=x, y=target)

# This gives the best possible estimator which further broken down in consequent steps
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)

# sklearn documentations allows result to be printed out in a dataframe to enable readability
gridsearch_results = pd.DataFrame(clf.cv_results_)
gridsearch_results.head(5)

# Select the most important details which are criterion,splitter,max_depth,
# min_samples_leaf,random_state and mean_test_score
tuned_par = gridsearch_results[['param_criterion','param_splitter','param_max_depth',
                                'param_min_samples_leaf','param_random_state','mean_test_score']]

# This is to sort the results in an ascending order, from observation of results the first 6 results only
# vary slightly based on a decrease of min_sample_leaf from a range of (1-2) and all 'entropy'
tuned_par.sort_values( by="mean_test_score",ascending=False).head(15)

# Separating the target variable
independent = working_data.values[:,0:15]
target = resampled_data.values[:, 16]

# Splitting the dataset into train and test (testing hold out)
independent_train, independent_test, target_train, target_test = train_test_split(
independent, target, test_size = 0.3, random_state = 100) 


independent,target,independent_train, independent_test, target_train, target_test

# Creating the classifier object based on the result of the gridsearch result
clf_entropy = DecisionTreeClassifier(criterion ="entropy",random_state = 91,max_depth=4,
                                  splitter="random",min_samples_leaf=19) 

# Performing training
clf_entropy.fit(independent_train, target_train)

# Function to make predictions// clf_object is the same as either clf_gini or clf_entropy
target_pred = clf_entropy.predict(independent_test)
print("Predicted values:")
print(target_pred)

# Function to calculate accuracy
print("Confusion Matrix: ",confusion_matrix(target_test, target_pred))
print ("Accuracy : ",accuracy_score(target_test,target_pred)*100)
print("Report : ", classification_report(target_test, target_pred))



print("Accuracy on training set: {:.3f}".format(clf_entropy.score(independent_train, target_train)))
print("Accuracy on test set: {:.3f}".format(clf_entropy.score(independent_test, target_test)))


from sklearn.tree import export_graphviz
export_graphviz(clf_entropy, out_file="tree.dot", class_names=["yes", "no"],
 feature_names=list(working_data.columns), impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
 dot_graph = f.read()
graphviz.Source(dot_graph)


from sklearn.metrics import roc_auc_score

auc_decision_tree = roc_auc_score(target_test, clf_entropy.predict_proba(independent_test)[:,1])
print(' ')
print('ROC_AUC is {} and accuracy rate is {}'.format(auc_decision_tree, clf_entropy.score(independent_test, target_test)))

from sklearn import metrics

disp = metrics.plot_confusion_matrix(clf_entropy, independent_test, target_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

important = pd.DataFrame(clf_entropy.feature_importances_)
with_labels = pd.DataFrame(list(working_data.columns))
feature_importance_table = pd.concat([with_labels,important],axis=1)
headers = ['Features',"Feature Importance"]
feature_importance_table.columns = headers
feature_importance_table = feature_importance_table.sort_values(by=['Feature Importance'],ascending=False)

print("Feature importances:\n{}".format(feature_importance_table))




# RANDOM FOREST MODEL

from sklearn.model_selection import RandomizedSearchCV
# The following hyperparameters would be iterated over to select the best possible combination
parameters = {'n_estimators':range(250,350),'max_depth':range(3,20),'max_features':range(3,20),
             'bootstrap':(True,False)}

rs = RandomizedSearchCV(RandomForestClassifier(), parameters, n_iter=3,cv=5,return_train_score=False)
rs.fit(X=x, y=target)

# sklearn documentations allows result to be printed out in a dataframe to enable readability
rs_results = pd.DataFrame(rs.cv_results_)
rs_results.head(5)

#Results from the output below:
# RandomForestClassifier(n_estimators=272,max_features=7,bootstrap=False,max_depth=14)
# RandomForestClassifier(n_estimators=319,max_features=9,bootstrap=False,max_depth=10)
# RandomForestClassifier(n_estimators=339,max_features=8,bootstrap=False,max_depth=3)

# Select the most important details
tuned_par = rs_results[['param_n_estimators','param_max_features','param_max_depth','param_bootstrap','mean_test_score']]

# This is to sort the results in an ascending order
tuned_par.sort_values( by="mean_test_score",ascending=False).head()

# The following outcomes would be iterated each to determine which parameters gives a good accuracy and less overfitting

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Classifier
clf1=RandomForestClassifier(n_estimators=272,max_features=7,bootstrap=False,max_depth=14)

# This is from the initial split in above cell also used for decision tree
independent,target,independent_train, independent_test, target_train, target_test

# Fit the model
tree = clf1.fit(independent_train,target_train)

# Prediction of outcomes
target_pred_randomforest =clf1.predict(independent_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred_randomforest))


print("Accuracy on training set: {:.3f}".format(clf1.score(independent_train, target_train)))
print("Accuracy on test set: {:.3f}".format(clf1.score(independent_test, target_test)))


# For the confusion matrix
from sklearn import metrics
display = metrics.plot_confusion_matrix(clf1, independent_test, target_test)
display.figure_.suptitle("Confusion Matrix for Random Forest")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Classifier
clf2=RandomForestClassifier(n_estimators=319,max_features=9,bootstrap=False,max_depth=10)

tree = clf2.fit(independent_train,target_train)

target_pred_randomforest =clf2.predict(independent_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred_randomforest))


print("Accuracy on training set: {:.3f}".format(clf2.score(independent_train, target_train)))
print("Accuracy on test set: {:.3f}".format(clf2.score(independent_test, target_test)))

from sklearn import metrics

display = metrics.plot_confusion_matrix(clf2, independent_test, target_test)
display.figure_.suptitle("Confusion Matrix for Random Forest")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Classifier
clf3=RandomForestClassifier(n_estimators=339,max_features=8,bootstrap=False,max_depth=3)

tree = clf3.fit(independent_train,target_train)

target_pred_randomforest =clf3.predict(independent_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred_randomforest))


print("Accuracy on training set: {:.3f}".format(clf3.score(independent_train, target_train)))
print("Accuracy on test set: {:.3f}".format(clf3.score(independent_test, target_test)))

display = metrics.plot_confusion_matrix(clf3, independent_test, target_test)
display.figure_.suptitle("Confusion Matrix for Random Forest")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

TO CHECK THE FEATURE IMPORTANCES IN THE RANDOM FOREST

# To check the features and effect on the tree

feature_imp =pd.Series(clf2.feature_importances_,index=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'previous',
       'poutcome']).sort_values(ascending=False)


# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

from sklearn import tree

fn=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'previous',
       'poutcome']
cn=['yes','no']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(clf2.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')



# The forest is really large, so lets try to extract the decision trees that makes up the random forest

fn=['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'previous',
       'poutcome']
cn=['yes','no']
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 15):
    tree.plot_tree(clf2.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_treess.png')

# Gradient Boosting Classifier

independent,target,independent_train, independent_test, target_train, target_test

from sklearn.ensemble import GradientBoostingClassifier


gbrt = GradientBoostingClassifier(random_state=10,learning_rate=0.3,n_estimators=100,subsample=0.8,criterion="mse",
                                 min_samples_leaf=11,max_depth=3,max_leaf_nodes=10)
gbrt.fit(independent_train, target_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(independent_train,target_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(independent_test, target_test)))




NEURAL NETWORK CLASSIFIER TRIALS

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)
mlp.fit(independent_train, target_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(independent_train, target_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(independent_test, target_test)))


#This neural network classifier are scaled sensitive as such there is a need to scale our dataset before using for prediction
from sklearn.neural_network import MLPClassifier

Calculated_mean = independent_train.mean(axis=0)
Calculated_Standard_deviation = independent_train.std(axis=0)
independent_train_scaled = (independent_train - Calculated_mean) / Calculated_Standard_deviation

# use THE SAME transformation (using training mean and std) on the test set
independent_test_scaled = (independent_test - Calculated_mean) / Calculated_Standard_deviation

# Building my neural network with the following steps:
neural_clf = MLPClassifier(random_state=42,max_iter=1000)
neural_clf.fit(independent_train_scaled, target_train)

print("Accuracy on training set: {:.3f}".format(neural_clf.score(independent_train_scaled, target_train)))
print("Accuracy on test set: {:.3f}".format(neural_clf.score(independent_test_scaled, target_test)))

GRAPHICAL COMPARISON BETWEEN ROC_AUC AND ACCURACY OF THE THREE MODELS

# Calculating the ROC_AUC of the three models
auc_dt= roc_auc_score(target_test, clf_gini.predict_proba(independent_test)[:,1])
auc_rf= roc_auc_score(target_test, clf.predict_proba(independent_test)[:,1])
auc_gbrt = roc_auc_score(target_test, gbrt.predict_proba(independent_test)[:,1])

# Calculating the accuracy of the three models
accuracy_dt = clf_gini.score(independent_test, target_test)
accuracy_rf = clf.score(independent_test, target_test)
accuracy_gbrt = gbrt.score(independent_test, target_test)

# Create a dataframe to house your above results and change index to column ('Models') to enable you plot a barchart representation
data = {'Models': ['Decision Tree','Random Forest','Gradient Boosting'],
        'ROC_AUC': [auc_dt,auc_rf,auc_gbrt],
        'Accuracy':[accuracy_dt,accuracy_rf,accuracy_gbrt]
       }

df = pd.DataFrame(data,columns=['Models','ROC_AUC','Accuracy'])
df.set_index('Models',inplace = True)

# Plot a bar chart showing the ROC_AUC and Accuracy of the three models
df.plot(kind='bar',color=['red','blue'],title= 'Comparing accuracy and ROC_AUC for each model',ylabel= 'ACCURACY and ROC_AUC',xlabel= 'Models used',figsize=(12,5))
plt.xticks(rotation=360)


from sklearn.metrics import roc_curve
target_test = pd.DataFrame(target_test)
target_test[0].map({'no':0, 'yes':1})

# calculate roc curve
dt_fpr, dt_tpr, dt_thresholds = roc_curve(target_test, clf_gini.predict_proba(independent_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds= roc_curve(target_test, clf.predict_proba(independent_test)[:,1])
gbrt_fpr, gbrt_tpr, gbrt_thresholds = roc_curve(target_test, gbrt.predict_proba(independent_test)[:,1])











from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

# Load the dataset
X_iris = load_iris().data
y_iris = load_iris().target

# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use a Support Vector Classifier with "rbf" kernel
svr = SVC(kernel="rbf")

# Create inner and outer strategies
inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)
nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv).mean(); print(nested_score)

from scipy.stats import wilcoxon
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Load the dataset
X = load_iris().data
y = load_iris().target

# Prepare models and select your CV method
model1 = ExtraTreesClassifier()
model2 = RandomForestClassifier()
kf = KFold(n_splits=20, random_state=42)

# Extract results for each model on the same folds
results_model1 = cross_val_score(model1, X, y, cv=kf)
results_model2 = cross_val_score(model2, X, y, cv=kf)

# Calculate p value
stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit'); p

