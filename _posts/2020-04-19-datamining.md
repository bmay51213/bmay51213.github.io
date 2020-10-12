---
title: "Echocardiograms and Survival Prediction After A Heart Attack"
date: 2020-04-19
classes: wide
header:
  image: "/images/datamining/a-screen-showing-an-echocardiogram.jpg"
excerpt: "Logistic Regression, SVG, Random Forest Classifier"
---
Coronary artery disease continues to be a significant cause of mortality.  The question remains is if higher risk patients for death could be identified and treated appropriately to avert mortality.

This dataset is from the UCI database describing characteristics on an echocardiogram after someone sustained a heart attack (MI - myocardial infarction) and their survival after one year.  Echocardiograms, or more simply an ultrasound of the heart, have many different measurements and the purpose of this project was to see if certain measurements or characteristics were predictive of survival at 1 year post-MI.

__Dataset URL:__

[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/echocardiogram)

__Source Cited:__

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Importing Dataset and Cleaning

```python
#Loading our Echocardiogram Dataset into a Pandas Dataframe.
df = pd.read_csv("echocardiogram.csv")

#Viewing our Dataframe
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survival</th>
      <th>alive</th>
      <th>age</th>
      <th>pericardialeffusion</th>
      <th>fractionalshortening</th>
      <th>epss</th>
      <th>lvdd</th>
      <th>wallmotion-score</th>
      <th>wallmotion-index</th>
      <th>mult</th>
      <th>name</th>
      <th>group</th>
      <th>aliveat1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.0</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>9.000</td>
      <td>4.600</td>
      <td>14.0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>name</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>0.0</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>0.380</td>
      <td>6.000</td>
      <td>4.100</td>
      <td>14.0</td>
      <td>1.700</td>
      <td>0.588</td>
      <td>name</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>4.000</td>
      <td>3.420</td>
      <td>14.0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>name</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>0.0</td>
      <td>0.253</td>
      <td>12.062</td>
      <td>4.603</td>
      <td>16.0</td>
      <td>1.450</td>
      <td>0.788</td>
      <td>name</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.0</td>
      <td>1.0</td>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.160</td>
      <td>22.000</td>
      <td>5.750</td>
      <td>18.0</td>
      <td>2.250</td>
      <td>0.571</td>
      <td>name</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>7.5</td>
      <td>1.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>0.240</td>
      <td>12.900</td>
      <td>4.720</td>
      <td>12.0</td>
      <td>1.000</td>
      <td>0.857</td>
      <td>name</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129</th>
      <td>41.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>0.0</td>
      <td>0.280</td>
      <td>5.400</td>
      <td>5.470</td>
      <td>11.0</td>
      <td>1.100</td>
      <td>0.714</td>
      <td>name</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>130</th>
      <td>36.0</td>
      <td>0.0</td>
      <td>69.0</td>
      <td>0.0</td>
      <td>0.200</td>
      <td>7.000</td>
      <td>5.050</td>
      <td>14.5</td>
      <td>1.210</td>
      <td>0.857</td>
      <td>name</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131</th>
      <td>22.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.140</td>
      <td>16.100</td>
      <td>4.360</td>
      <td>15.0</td>
      <td>1.360</td>
      <td>0.786</td>
      <td>name</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>132</th>
      <td>20.0</td>
      <td>0.0</td>
      <td>62.0</td>
      <td>0.0</td>
      <td>0.150</td>
      <td>0.000</td>
      <td>4.510</td>
      <td>15.5</td>
      <td>1.409</td>
      <td>0.786</td>
      <td>name</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>133 rows × 13 columns</p>
</div>


There are 133 data points and 13 variables including the target variable.

### Description of the Variables in the Dataframe ###

__Survival (in Months):__  Numerical

__Alive:__ Categorical (0 or 1)

__Age in Years When Heart Attack Occurred:__ Numerical

__Pericardial Effusion:__  Categorical (0 or 1)

__Fractional Shortening Measurement:__ Numerical (Measurement of Contractility of Heart, higher the better)

__EPSS (E Point Septal Separation, the lower the better):__ Numerical

__Left Ventricular Diastolic Dimension (LVDD):__ Numerical

__Wall Motion Score (Score Of How Walls of Heart Move):__ Numerical (Integers)

__Wall Motion Index (Score Divided By Number of Segments Seen - Usually 12-13):__ Numerical

__Mult, Name, and Group:__ All are case identifiers and not pertinent to the Analysis

__Aliveat1:__  Categorical (Whether or Not Person Was Alive At One Year -- __Target__)

The dataset was cleaned of the case identifiers.

```python
#Data Cleaning

#Dropping the mult, name, and group variables
df.drop(['mult', 'name', 'group'], axis=1, inplace = True)

#Since Alive At 1 Year is Our Target Variable and was derived form survival and alive, I will drop the survival and alive
#columns
df.drop(['survival', 'alive'], axis = 1, inplace = True)

#Since wallmotion index is also derived from the wall-motion score, I'm also going to drop the wallmotion-score column
df.drop(['wallmotion-score'], axis = 1, inplace = True)

#Viewing Dataframe Post Cleaning
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>pericardialeffusion</th>
      <th>fractionalshortening</th>
      <th>epss</th>
      <th>lvdd</th>
      <th>wallmotion-index</th>
      <th>aliveat1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>9.000</td>
      <td>4.600</td>
      <td>1.000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72.0</td>
      <td>0.0</td>
      <td>0.380</td>
      <td>6.000</td>
      <td>4.100</td>
      <td>1.700</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>4.000</td>
      <td>3.420</td>
      <td>1.000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>0.253</td>
      <td>12.062</td>
      <td>4.603</td>
      <td>1.450</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.160</td>
      <td>22.000</td>
      <td>5.750</td>
      <td>2.250</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>64.0</td>
      <td>0.0</td>
      <td>0.240</td>
      <td>12.900</td>
      <td>4.720</td>
      <td>1.000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>129</th>
      <td>64.0</td>
      <td>0.0</td>
      <td>0.280</td>
      <td>5.400</td>
      <td>5.470</td>
      <td>1.100</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>130</th>
      <td>69.0</td>
      <td>0.0</td>
      <td>0.200</td>
      <td>7.000</td>
      <td>5.050</td>
      <td>1.210</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>131</th>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.140</td>
      <td>16.100</td>
      <td>4.360</td>
      <td>1.360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>132</th>
      <td>62.0</td>
      <td>0.0</td>
      <td>0.150</td>
      <td>0.000</td>
      <td>4.510</td>
      <td>1.409</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>133 rows × 7 columns</p>
</div>



There are null values in this dataset and will be removed for analysis.


```python
#Dropping null values
df.dropna(inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>pericardialeffusion</th>
      <th>fractionalshortening</th>
      <th>epss</th>
      <th>lvdd</th>
      <th>wallmotion-index</th>
      <th>aliveat1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>9.000</td>
      <td>4.600</td>
      <td>1.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72.0</td>
      <td>0.0</td>
      <td>0.380</td>
      <td>6.000</td>
      <td>4.100</td>
      <td>1.70</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>4.000</td>
      <td>3.420</td>
      <td>1.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>0.0</td>
      <td>0.253</td>
      <td>12.062</td>
      <td>4.603</td>
      <td>1.45</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.160</td>
      <td>22.000</td>
      <td>5.750</td>
      <td>2.25</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>63.0</td>
      <td>0.0</td>
      <td>0.300</td>
      <td>6.900</td>
      <td>3.520</td>
      <td>1.51</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>106</th>
      <td>59.0</td>
      <td>0.0</td>
      <td>0.170</td>
      <td>14.300</td>
      <td>5.490</td>
      <td>1.50</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.228</td>
      <td>9.700</td>
      <td>4.290</td>
      <td>1.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>78.0</td>
      <td>0.0</td>
      <td>0.230</td>
      <td>40.000</td>
      <td>6.230</td>
      <td>1.40</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>110</th>
      <td>62.0</td>
      <td>0.0</td>
      <td>0.260</td>
      <td>7.600</td>
      <td>4.420</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 7 columns</p>
</div>


The variables were recoded into Python to represent the appropriate categorical values.  Columns were also renamed for simplicity.

## Data Exploration and Analysis

A descriptive analysis was done on both the numerical and categorical variables.


```python
print('Description of Data')
df.describe()
```

    Description of Data





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fractionalshortening</th>
      <th>epss</th>
      <th>lvdd</th>
      <th>wallmotion-index</th>
      <th>pericardialeffusion_1.0</th>
      <th>aliveat1_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.419355</td>
      <td>0.218452</td>
      <td>12.307387</td>
      <td>4.817129</td>
      <td>1.406403</td>
      <td>0.177419</td>
      <td>0.290323</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.639498</td>
      <td>0.106001</td>
      <td>7.305048</td>
      <td>0.774996</td>
      <td>0.445460</td>
      <td>0.385142</td>
      <td>0.457617</td>
    </tr>
    <tr>
      <th>min</th>
      <td>46.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>3.420000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>58.250000</td>
      <td>0.150000</td>
      <td>8.125000</td>
      <td>4.290000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>62.000000</td>
      <td>0.218500</td>
      <td>11.000000</td>
      <td>4.601500</td>
      <td>1.321500</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>0.267500</td>
      <td>15.900000</td>
      <td>5.422500</td>
      <td>1.625000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>86.000000</td>
      <td>0.610000</td>
      <td>40.000000</td>
      <td>6.730000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The youngest person in the analysis was 46 years old which is young for someone to sustain a heart attack.  The oldest was 86 years old and the average age was 64 years old.

The left ventricular diastolic dimension mean was 4.8 which is within normal range.  The largest was 6.73 cm which is very large.  The minimum was 3.42 though there is less of a concern with the lower LVIDD as there is with elevated ones.

The higher the level of fractional shortening is the better.  The minimum was very low at 0.01 and the largest was 0.61.  Mean values are around 0.21.


```python
#Checking Variable Distributions

#Setting Figure Size
plt.rcParams['figure.figsize']= (20,10)

#Initiating Our Subplots
fig, axes = plt.subplots(nrows = 2, ncols = 3, constrained_layout = True)

#Identifying Numerical Features of Interest (Age, Fractional Shortening, EPSS, LVDD, Wall-Motion Score, and Wall-Motion
#Index)
num_features = ['age', 'fractionalshortening', 'epss', 'lvdd', 'wallmotion-index']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts']

#Histogram Creation
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(df[num_features[idx]], bins = 100)
    ax.set_xlabel(xaxes[idx], fontsize = 20)
    ax.set_ylabel(yaxes[idx], fontsize = 20)
    ax.tick_params(axis='both', labelsize = 15)
plt.show()
```


![png](/images/datamining/output_18_1.png)


Age at heart attack and fractional shortening appear to be normally distributed.  The variables of epss, lvdd, and wallmotionindex appear to be positively skewed.  


```python
#Checking Categorical Variables Explicitly
df.describe(include=['O'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pericardialeffusion</th>
      <th>aliveat1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>62.0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>top</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>51.0</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>



The majority of people in the study group did not have a pericardial effusion.  Those with pericardial effusions are higher cardiac risk so this is a favorable thing.

The majority of people in the study were alive after their heart attack.


![png](/images/datamining/output_22_0.png)



![png](/images/datamining/output_22_1.png)


The numerical variables were analyzed for high levels of correlation (>0.90) and plotted using a correlation plot with the seaborn package.


```python
#Checking for correlation
%matplotlib inline
plt.rcParams['figure.figsize'] = (10,10)

#Importing Seaborn
import seaborn as sns

#Checking Correlation Matrix
corr_matrix = df.corr()

#Visualizing Correlation Matrix As Heatmap
sns.heatmap(corr_matrix)
plt.show()
```


![png](/images/datamining/output_24_0.png)


Based on the above correlation plot, EPSS and LVDD appear to be positively correlated as do EPSS and wallmotion-index.  LVDD and wallmotion index appear to be positively correlated as well.  Age and fractional shortening appear to be slightly negatively correlated.


```python
#Checking for Highly Correlated Features For Validation
corr_matrix = df.corr()
corr_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fractionalshortening</th>
      <th>epss</th>
      <th>lvdd</th>
      <th>wallmotion-index</th>
      <th>pericardialeffusion_1.0</th>
      <th>aliveat1_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>-0.116029</td>
      <td>0.079404</td>
      <td>0.199105</td>
      <td>0.086844</td>
      <td>0.021614</td>
      <td>0.354324</td>
    </tr>
    <tr>
      <th>fractionalshortening</th>
      <td>-0.116029</td>
      <td>1.000000</td>
      <td>-0.399324</td>
      <td>-0.369920</td>
      <td>-0.327916</td>
      <td>-0.005207</td>
      <td>-0.269395</td>
    </tr>
    <tr>
      <th>epss</th>
      <td>0.079404</td>
      <td>-0.399324</td>
      <td>1.000000</td>
      <td>0.569668</td>
      <td>0.442790</td>
      <td>-0.031571</td>
      <td>0.271188</td>
    </tr>
    <tr>
      <th>lvdd</th>
      <td>0.199105</td>
      <td>-0.369920</td>
      <td>0.569668</td>
      <td>1.000000</td>
      <td>0.339077</td>
      <td>-0.029571</td>
      <td>0.268640</td>
    </tr>
    <tr>
      <th>wallmotion-index</th>
      <td>0.086844</td>
      <td>-0.327916</td>
      <td>0.442790</td>
      <td>0.339077</td>
      <td>1.000000</td>
      <td>0.148064</td>
      <td>0.445502</td>
    </tr>
    <tr>
      <th>pericardialeffusion_1.0</th>
      <td>0.021614</td>
      <td>-0.005207</td>
      <td>-0.031571</td>
      <td>-0.029571</td>
      <td>0.148064</td>
      <td>1.000000</td>
      <td>0.168025</td>
    </tr>
    <tr>
      <th>aliveat1_1.0</th>
      <td>0.354324</td>
      <td>-0.269395</td>
      <td>0.271188</td>
      <td>0.268640</td>
      <td>0.445502</td>
      <td>0.168025</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Given these variables, none are significantly highly correlated that could prove problems in future
data analysis (correlation value of 0.95 or above).

Several variables were compared via scatterplot to look for interesting relationships.


#Scatterplot for EPSS and Left Ventricular Diastolic Dimension
plt.rcParams['figure.figsize'] = (6,5)
plt.scatter(epss, lvdd)
plt.xlabel('E Point Septal Separation')
plt.ylabel('Left Ventricular Diastolic Dimension (cm)')
plt.title('EPSS vs. Left Ventricular Diastolic Dimension')
plt.show()
```


![png](/images/datamining/output_29_0.png)



```python
#Plotting EPSS vs. Wall Motion Index
plt.rcParams['figure.figsize'] = (6,5)
plt.scatter(epss, wmi)
plt.xlabel('E Point Septal Separation')
plt.ylabel('Wall Motion Index')
plt.title('E Point Septal Separation vs. Wall Motion Index')
plt.show()
```


![png](/images/datamining/output_30_0.png)



```python
#Plotting EPSS vs. Fractional Shortening
plt.rcParams['figure.figsize'] = (6,5)
plt.scatter(epss, fracshort)
plt.xlabel('E Point Septal Separation')
plt.ylabel('Fractional Shortening')
plt.title('E Point Septal Separation Vs. Fractional Shortening')
plt.show()
```


![png](/images/datamining/output_31_0.png)



```python
#Plotting LVDD vs. Fractional shortening
plt.rcParams['figure.figsize'] = (6,5)
plt.scatter(lvdd, fracshort)
plt.xlabel('Left Ventricular Diastolic Dimension (cm)')
plt.ylabel('Fractional Shortening')
plt.title('Left Ventricular Diastolic Dimension and Fractional Shortening')
plt.show()
```


![png](/images/datamining/output_32_0.png)


There appeared to be a weak positive correlation between EPSS and LVDD as well as a weak negative correlation between LVDD and Fractional Shortening.


```python
#Plotting stacked bar charts for our categorical variables and survival
%matplotlib inline
plt.rcParams['figure.figsize'] = (6,5)

#Subplots
fig, axes = plt.subplots()

#Feeding Data Into Visualizer Based on Percardial Effusion Presence
pe_survived = df.replace({'aliveat1': {1: 'Alive', 0: 'Deceased'}})[df['aliveat1']==1]['pericardialeffusion'].value_counts()
pe_deceased = df.replace({'aliveat1': {1: 'Alive', 0: 'Deceased'}})[df['aliveat1']==0]['pericardialeffusion'].value_counts()
pe_deceased = pe_deceased.reindex(index=pe_survived.index)

#Making Bar Plot
p1 = plt.bar(pe_survived.index, pe_survived.values)
p2 = plt.bar(pe_deceased.index, pe_deceased.values, bottom = pe_survived.values)
plt.xticks([0,1], ['Absent', 'Present'])
plt.title('Pericardial Effusion Presence and Survival')
plt.ylabel('Counts')
plt.tick_params(axis = 'both')
plt.legend((p1[0], p2[0]), ('Alive', 'Deceased'))

plt.show()
```


![png](/images/datamining/output_34_0.png)


Looking at this stacked bar chart, for those who did not have a pericardial effusion, approximately 30% were alive and 70% were deceased.  For those who did have a pericardial effusion, approximately 50% survived and 50% died.  I suspect that there may be other variables influencing this.  Pericardial effusions are an uncommon entity and usually increase the risk of death so it is counter-intuitive that those without a pericardial effusion would be dead at 1 year.

### Dimensionality Reduction and Fitting of Machine Learning Algorithms

For eventual fitting of our machine learning algorithms, pandas was used to get dummy variables for the categorical values in the dataframe.

Then, feature selection was performed to simplify variables for the machine learning algorithms.

```python
#Using Feature Selection to Eliminate Variables That Are Not Useful

#Setting our Target: Alive at 1 year
target = df['aliveat1_1.0']

#Setting all features
features = df[['age', 'pericardialeffusion_1.0','fractionalshortening', 'epss', 'lvdd', 'wallmotion-index']]

#Loading Libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile

#Standardizing Our Numerical Features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#Select features with 75th Percentile
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features_standardized, target)

#Show Results
print("Original Number of Features:", features.shape[1])
print("Reduced Number of Features:", features_kbest.shape[1])

#Getting the names of the columns that were kept
fvalue_selector.get_support()
```

    Original Number of Features: 6
    Reduced Number of Features: 4

    array([ True, False,  True,  True, False,  True])



Based on this Boolean, it kept the variables Age, Fractional Shortening, EPSS, and Wall-Motion Index using 75th Percentile and LVDD and Pericardial Effusion should be dropped.

### Logistic Regression Feature Selection and Model Fitting


```python
#Trying RFECV with Logistic Regression to Recursively Eliminate Features of All Variables Including Numerical and
#Categorical Variables Using Neg Mean Squared Error as Scoring Metric

#Importing Our Packages
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import sklearn.linear_model as lm

#Setting Our Regression and using logistic regression since this is a binary predictor
regression = lm.LogisticRegression()

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Neg Mean Squared Error
selector = RFECV(estimator=regression, step=1, cv=StratifiedKFold(10), scoring='neg_mean_squared_error')
selector.fit(features_standardized,target)
print("Optimal Number of Features: %d" % selector.n_features_)

#Visualizing which features were kept
selector.get_support()
```

    Optimal Number of Features: 1

    array([False, False, False, False, False,  True])



Using negative mean squared error as our scoring metric with logistic regression, it appears that with this feature selection, only wall-motion index should be included.


```python
#Trying RFECV with Logistic Regressionto Recursively Eliminate Features of All Variables Including Numerical and
#Categorical Variables Using Accuracy as my scoring metric

#Setting Our Regression and using logistic regression since this is a binary predictor
regression = lm.LogisticRegression()

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Neg Mean Squared Error
selector = RFECV(estimator=regression, step=1, cv=StratifiedKFold(10), scoring='accuracy')
selector.fit(features_standardized,target)

print("Optimal Number of Features: %d" % selector.n_features_)

#Visualizing which features were kept
selector.get_support()
```

    Optimal Number of Features: 1

    array([False, False, False, False, False,  True])



Likewise, using accuracy as our scoring metric with logistic regression, it appears that with this round of feature selection, only wall motion index should be included.  It does not appear that in this case that using negative mean squared error or accuracy would change the variables that were selected.

The model was then fit using logistic regression using all variables to begin.

```python
#Using Stratified K Fold Cross Validation, We Will Run a Logistic Regression Model Using All Features

#Importing our Packages
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Creating Standardizer
standardizer = StandardScaler()

#Creating Logistic Regression Object
logit = LogisticRegression()

#Creating K Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, logit)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features, target, cv = kf, scoring = "accuracy", n_jobs = -1)
cv_results

#Evaluating our Metrics of Our Logistic Regression Classifier Using ALl Features

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC


#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(logit, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance?
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting Size of Figure/Font Size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(logit, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()

#ROC and AUC: Instantiating the Visualizer
visualizer2 = ROCAUC(logit)
visualizer2.fit(features_train, target_train)
visualizer2.score(features_test, target_test)
g = visualizer2.poof()
```


![png](/images/datamining/output_46_0.png)



![png](/images/datamining/output_46_1.png)



![png](/images/datamining/output_46_2.png)


Then, the model was run using our feature selected variables.

```python
standardizer = StandardScaler()

#Creating Logistic Regression Object
logit = LogisticRegression()

#Creating K Fold Cross Validation

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split

features_train, features_test, target_train, target_test = train_test_split(features3, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets

features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, logit)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features3, target, cv = kf, scoring = "accuracy", n_jobs = -1)
cv_results

#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(logit, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance?
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting size of figure and font size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(logit, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()

#ROC and AUC: Instantiating the Visualizer
visualizer2 = ROCAUC(logit)
visualizer2.fit(features_train, target_train)
visualizer2.score(features_test, target_test)
g = visualizer2.poof()
```


![png](/images/datamining/output_47_0.png)



![png](/images/datamining/output_47_1.png)



![png](/images/datamining/output_47_2.png)


### Random Forest Feature Selection and Model Fitting


```python
#Using Random Forest Classifier as our Model with Negative Mean Squared Error as Scoring Metric

#Importing Package
from sklearn.ensemble import RandomForestClassifier

#Setting Up Our Random Forest Classifier to Call in our RFECV Function
rfc = RandomForestClassifier(n_jobs=-1)

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Neg Mean Squared Error
selector = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='neg_mean_squared_error')
selector.fit(features_standardized,target)

print("Optimal Number of Features: %d" % selector.n_features_)

selector.get_support()
```

    Optimal Number of Features: 4
    array([ True, False, False,  True,  True,  True])




```python
#Using Random Forest Classifier as our Model with Accuracy as Scoring Metric

#Setting Up Our Random Forest Classifier to Call in our RFECV Function
rfc = RandomForestClassifier(n_jobs=-1)

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Accuracy
selector = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
selector.fit(features_standardized,target)

print("Optimal Number of Features: %d" % selector.n_features_)

#Visualizing which features were kept
selector.get_support()
```

    Optimal Number of Features: 3
    array([ True, False, False,  True, False,  True])



Depending on which time the data is run, some of the Random Forest Classifier models recommend keeping all variables or excluding the pericardial effusion variable, but this changes from each iteration.

The Random Forest Classifier was fit using all variables to begin.

```python
#Using Stratified K Fold Cross Validation, We Will Run a Random Forest Classification Model Using All Features

#Creating Standardizer
standardizer = StandardScaler()

#Creating Random Forest Object
rfc = RandomForestClassifier(random_state = 111, n_jobs = -1, class_weight = "balanced")

#Creating K Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, rfc)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features, target, cv = kf, scoring = "accuracy", n_jobs = -1)

#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(rfc, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting size of figure and font size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(rfc, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()

#ROC and AUC and Instantiating the Visualizer
visualizer2 = ROCAUC(rfc)
visualizer2.fit(features_train, target_train)
visualizer2.score(features_test, target_test)
g = visualizer2.poof()
```


![png](/images/datamining/output_53_0.png)



![png](/images/datamining/output_53_1.png)



![png](/images/datamining/output_53_2.png)



Then, the Random Forest Classifier was run using the feature selected variables.

```python
standardizer = StandardScaler()

#Creating Random Forest Regression Object
rfc = RandomForestClassifier(random_state = 111, n_jobs = -1, class_weight = "balanced")

#Creating K Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split
features_train, features_test, target_train, target_test = train_test_split(features2, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, rfc)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features2, target, cv = kf, scoring = "accuracy", n_jobs = -1)

#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(rfc, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting size of figure and font size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(rfc, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()

#ROC and AUC and Instantiating the Visualizer
visualizer2 = ROCAUC(rfc)
visualizer2.fit(features_train, target_train)
visualizer2.score(features_test, target_test)
g = visualizer2.poof()
```


![png](/images/datamining/output_54_0.png)



![png](/images/datamining/output_54_1.png)



![png](/images/datamining/output_54_2.png)


### SVM Feature Selection and Model Fitting


```python
#Trying RFECV with SVM (Support Vector Machine) to Recursively Eliminate Features of All Variables Including Numerical and
#Categorical Variables using accuracy as my scoring metric

#Importing Our Packages
from sklearn.svm import SVC

#Setting Up SVM model
clf = SVC(kernel='linear')

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Accuracy
selector = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10), scoring='accuracy')
selector.fit(features_standardized,target)

print("Optimal Number of Features: %d" % selector.n_features_)

#Visualizing which features were kept
selector.get_support()
```

    Optimal Number of Features: 3
    array([ True,  True, False, False, False,  True])




```python
#Trying RFECV with SVM (Support Vector Machine) to Recursively Eliminate Features of All Variables Including Numerical and
#Categorical Variables using Negative Mean Squared Error as Scoring metric

#Setting Up SVM model
clf = SVC(kernel='linear')

#Setting Our Selector for Stratified K Fold Cross validation of 10 and Using Neg Mean Squared Error
selector = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10), scoring='neg_mean_squared_error')
selector.fit(features_standardized,target)

print("Optimal Number of Features: %d" % selector.n_features_)

#Visualizing which features were kept
selector.get_support()
```

    Optimal Number of Features: 3
    array([ True,  True, False, False, False,  True])



Using Support Vector Machine, feature selection identified using age, pericardialeffusion_1.0, and, wallmotion-index for our model.

First, the model was fit using all variables.

```python

clf = SVC(kernel='linear')

#Creating K Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, clf)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features, target, cv = kf, scoring = "accuracy", n_jobs = -1)


#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(clf, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting size of figure and font size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(clf, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()
```


![png](/images/datamining/output_59_0.png)



![png](/images/datamining/output_59_1.png)


Then, the SVM model was run using the feature selected variables.

```python

clf = SVC(kernel='linear')

#Creating K Fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 1)

#Doing Training/Test Split
features_train, features_test, target_train, target_test = train_test_split(features4, target, test_size=0.3, random_state = 1)

#Fitting Standardizer
standardizer.fit(features_train)

#Applying to both training and test sets
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

#Creating Pipeline
pipeline = make_pipeline(standardizer, clf)

#Do K Fold Cross-Validation
cv_results = cross_val_score(pipeline, features4, target, cv = kf, scoring = "accuracy", n_jobs = -1)

#Confusion Matrix Visualizer
classes = ['Not Survived', 'Survived']
cm = ConfusionMatrix(clf, classes = classes, percent = False)

#Fitting the passed model
cm.fit(features_train, target_train)

#Create confusion matrix
cm.score(features_test, target_test)

#Change fontsize of labels in figure
for label in cm.ax.texts:
    label.set_size(20)

#Checking model performance
cm.poof()

#Getting Precision, Recall, and F1 Score and ROC Curve and Setting size of figure and font size
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['font.size'] = 20

#Instantiate visualizer
visualizer1 = ClassificationReport(clf, classes = classes)

#Fit training data to visualizer
visualizer1.fit(features_train, target_train)

#Evaluating model on the test data
visualizer1.score(features_test, target_test)
g = visualizer1.poof()
```


![png](/images/datamining/output_60_0.png)



![png](/images/datamining/output_60_1.png)


One limitation of this dataset is the small sample size after we removed the null values.  Another limitation of the dataset is that the deceased proportion of individuals was nearly two times those who survived in the dataset which will skew the results.  The models tend to be better here at predicting not survived vs. survived. This is likely explained by the fact that our dataset had 2 times the amount of people deceased vs. survived so there was a smaller amount of data to help predict the survived categories.__

__I used three different methods, Logistic Regression, Random Forest Classification, and Support Vector Machine Classification.  The best performing model was Logistic Regression followed closely by Support Vector Machines.  The Random Forest models performed the worst.  Feature selection clearly improved the results of all of the models based on their Precision, Recall, and F1 scores.

__To view more specific coding and visualizations, please refer to my GitHub Repository.__

```
