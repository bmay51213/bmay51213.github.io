---
title: "Detection of Breast Cancer In Biopsy Specimens Using Machine Learning"
date: 2020-10-02
classes: wide
header:
excerpt: "Logistic Regression, KNN, Random Forest"
---

![png](/images/breastcancer/1_yjsLGG-U9km84AvWLLmK8A.png)

Breast cancer continues to be a significant culprit of morbidity and mortality, even today with our current medical advances.  The data that this dataset consists of was actually published in a medical journal article and data was collected at the University of Wisconsin (Wolberg, M.D. et al.¸1995). While this source is somewhat dated, my main goal with this project is to demonstrate that machine learning algorithms could be used to assist in cancer detection as adjunct to physician expertise.

__Problem Background:__

It is no secret that our healthcare is cumbersome, overpriced, and our outcomes are unsatisfactory when compared to other peer countries.  In a data-driven world, with the vast amounts of personal health information and data available in electronic health records, the sky is the limit.  Could machine learning algorithms be used to improve diagnosis, save lives, and prevent suffering?  

Breast cancer is the second most common cause of death due to cancer in women.  Using mammograms, women are screened for breast cancer.  Suspicious areas identified radiologically are then biopsied using a wide variety of techniques.  One method is using a fine needle aspirate in which cells from the area in question are extracted and examined under the microscope.  Even with expert analysis, diagnosis of these areas can be challenging.

Therefore, the question remains, can machine learning help us predict malignant tumors as an adjunct to physician expertise?

__Dataset Description:__

The data is from the University of California - Irvine Machine Learning Repository and can be found here:  

[https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

This data was collected at the University of Wisconsin in 1995.  The file is in a csv format in which microscopic images of Fine Needle Aspirates (a type of biopsy) of suspicious breast tissue was digitized.  There are a total of 32 variables with 570 subjects.  They were examining suspicious masses in those without evidence of metastasis (distant spread of cancer to other parts of the body).

After data cleaning, exploration, and analysis, the data will be fit and predictive ability evaluated using Logistic Regression, K Nearest Neighbors, and Random Forest machine learning algorithms.

# Data Cleaning, Exploration, and Analysis

The dataframe is included below:

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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.30010</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.380</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.08690</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.990</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.19740</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.570</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.24140</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.910</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.19800</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.540</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
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
      <th>564</th>
      <td>926424</td>
      <td>M</td>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>...</td>
      <td>25.450</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
    </tr>
    <tr>
      <th>565</th>
      <td>926682</td>
      <td>M</td>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>...</td>
      <td>23.690</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
    </tr>
    <tr>
      <th>566</th>
      <td>926954</td>
      <td>M</td>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>...</td>
      <td>18.980</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
    </tr>
    <tr>
      <th>567</th>
      <td>927241</td>
      <td>M</td>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>...</td>
      <td>25.740</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
    </tr>
    <tr>
      <th>568</th>
      <td>92751</td>
      <td>B</td>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>9.456</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 32 columns</p>
</div>


### Description of the Variables

__Id:__ Number signifying unique samples (Integer)

__Diagnosis:__ M for Malignant, B for Benign (Categorical) – the Target

__Radius_mean:__ Mean of distance from center to points on the perimeter of tumor cell, cell size measure (Float)

__Texture_mean:__ Mean of grey-scale values of image (Float)

__Perimeter_mean:__ Expression of both cell size and shape (Float)

__Area_mean:__ Mean area of cell size (Float)

__Smoothness:__ Mean of cell smoothness and shape (Float)

__Compactness:__ Mean of cell compactness and shape (Float)

__Concavity_mean:__  Mean of cell concavity of image (Float)

__Concave points_mean:__ Mean of concave points (Float)

__Symmetry_mean:__ Mean of cell symmetry (Float)

__Fractal_dimension_mean:__ Mean of fractal dimension, measure of cell shape (Float)

__Radius_se:__ Standard error of distance from center to points on the perimeter of tumor cell, cell size measure (Float)

__Texture_se:__ Standard error of grey-scale values of image (Float)

__Perimeter_se:__ Standard error of both cell size and shape (Float)

__Area_se:__ Standard error of cell size area (Float)

__Smoothness_se:__ Standard error of cell smoothness (Float)

__Compactness_se:__ Standard error of cell compactness (Float)

__Concavity_se:__  Standard error of cell concavity of image (Float)

__Concave points_se:__ Standard error of concave points (Float)

__Symmetry_se:__ Standard error of cell symmetry (Float)

__Fractal_dimension_se:__ Standard error of fractal dimension (Float)

__Radius_worst:__ Worst measurement of distance from center to points on the perimeter of tumor cell, cell size measure (Float)

__Texture_worst:__ Worst measurement of grey-scale values of image (Float)

__Perimeter_worst:__ Worst measurement of both cell size and shape (Float)

__Area_worst:__ Worst measurement of cell size area (Float)

__Smoothness_worst:__ Worst measurement of cell smoothness (Float)

__Compactness_worst:__ Worst measurement of cell compactness (Float)

__Concavity_worst:__  Worst measurement of cell concavity of image (Float)

__Concave points_worst:__ Worst measurement of concave points (Float)

__Symmetry_worst:__ Worst measurement of cell symmetry (Float)

__Fractal_dimension_worst:__ Worst measurement of fractal dimension (Float)


The target variable is categorical and is coded as M for malignant and B for benign.  There are no missing data in the CSV file.  There are case identifiers but other demographic information such as age, co-morbidities, and family history are not available.  All of the variables were coded correctly into Python.

The variables examined include measurements of digitized images on both cell size and shape.  These were already identified as either benign or malignant.  The data aims to determine if these specific characteristics can be used to predict whether a tumor was malignant or benign.

The case identifiers and time variables were dropped from the dataset and descriptive analysis was performed of the numeric values.

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
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concavepoints_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concavepoints_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>

Descriptive statistics of the categorical variables were calculated.

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
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>B</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>357</td>
    </tr>
  </tbody>
</table>
</div>

The most common value was benign diagnoses of the categorical target variable.  Highly correlated variables were then identified.


```python
corr_matrix = df.corr()

#Selecting Upper Triangle of Correlation Matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

#Find Index of Feature Columns with Correlations >0.90
to_drop = [column for column in upper.columns if any (upper[column] >0.95)]
to_drop
```

    ['perimeter_mean',
     'area_mean',
     'perimeter_se',
     'area_se',
     'radius_worst',
     'perimeter_worst',
     'area_worst']


Given the high rates of correlation of the above variables, we will drop them from the dataframe.  With a correlation >0.95, these are likely redundant and may introduce instability in our final algorithms.


There were several outliers identified using the interquartile range.  There were various categories where the outliers were identified with a wide variety of variables.  The most common variable where it was noted was compactness.  The majority of these outliers were mostly identified as malignant though there were some benign diagnoses with outliers in the compactness variables.

```python
#Visualizing Outliers Using Boxplots

plt.figure(figsize=[100,100])

fig,axs = plt.subplots(9,3, figsize=(50, 50))

axs[0,0].boxplot(df['radius_mean'])
axs[0,0].set_title('Boxplot of Mean Radius')
axs[0,1].boxplot(df['radius_se'])
axs[0,1].set_title('Boxplot of Radius S.E.')
axs[0,2].boxplot(df['compactness_mean'])
axs[0,2].set_title('Boxplot of Mean Compactness')
axs[1,0].boxplot(df['compactness_se'])
axs[1,0].set_title('Boxplot of Compactness S.E.')
axs[1,1].boxplot(df['compactness_worst'])
axs[1,1].set_title('Boxplot of Worst Compactness')
axs[1,2].boxplot(df['smoothness_mean'])
axs[1,2].set_title('Boxplot of Mean Smoothness')
axs[2,0].boxplot(df['smoothness_se'])
axs[2,0].set_title('Boxplot of Smoothness S.E.')
axs[2,1].boxplot(df['smoothness_worst'])
axs[2,1].set_title('Boxplot of Worst Smoothness')
axs[2,2].boxplot(df['texture_mean'])
axs[2,2].set_title('Boxplot of Mean Texture')
axs[3,0].boxplot(df['texture_se'])
axs[3,0].set_title('Boxplot of Texture S.E.')
axs[3,1].boxplot(df['texture_worst'])
axs[3,1].set_title('Boxplot of Worst Texture')
axs[3,2].boxplot(df['compactness_mean'])
axs[3,2].set_title('Boxplot of Mean Compactness')
axs[4,0].boxplot(df['compactness_se'])
axs[4,0].set_title('Boxplot of Compactness S.E.')
axs[4,1].boxplot(df['compactness_worst'])
axs[4,1].set_title('Boxplot of Worst Compactness')
axs[4,2].boxplot(df['concavity_mean'])
axs[4,2].set_title('Boxplot of Mean Concavity')
axs[5,0].boxplot(df['concavity_se'])
axs[5,0].set_title('Boxplot of Concavity S.E.')
axs[5,1].boxplot(df['concavity_worst'])
axs[5,1].set_title('Boxplot of Worst Concavity')
axs[5,2].boxplot(df['symmetry_mean'])
axs[5,2].set_title('Boxplot of Mean Symmetry')
axs[6,0].boxplot(df['symmetry_se'])
axs[6,0].set_title('Boxplot of Symmetry S.E.')
axs[6,1].boxplot(df['symmetry_worst'])
axs[6,1].set_title('Boxplot of Worst Symmetry')
axs[6,2].boxplot(df['concavepoints_se'])
axs[6,2].set_title('Boxplot of Concave Points S.E.')
axs[7,0].boxplot(df['concavepoints_worst'])
axs[7,0].set_title('Boxplot of Worst Concave Points')
axs[7,1].boxplot(df['fractal_dimension_mean'])
axs[7,1].set_title('Boxplot of Mean Fractal Dimension')
axs[7,2].boxplot(df['fractal_dimension_se'])
axs[7,2].set_title('Boxplot of Fractal Dimension S.E.')
axs[8,0].boxplot(df['fractal_dimension_worst'])
axs[8,0].set_title('Boxplot of Worst Fractal Dimension')

```

![png](/images/breastcancer/output_18_2.png)


Many of the variables had outliers on the positive end of their measurement spectrum.  Given that these outliers could be predictive of being malignant, these will not be removed from the dataset.


```python
#Visualizing Distributions of Numerical Data

#Setting Figure Size
plt.figure(figsize=[100,100])

f,a = plt.subplots(8,3, figsize=(40,40))
plt.subplots_adjust(hspace=0.25, wspace = 0.25)

a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(df.iloc[:,idx], bins=100)
    ax.set_title(df.columns[idx])
    ax.set_ylabel('Counts')
plt.show()
```

![png](/images/breastcancer/output_20_1.png)


Many of these variables are positively skewed.  The only variables that appear to be mostly normally distributed are symmetry_mean and possible fractal_dimension_mean.  Concave_points_worst actually appears to be bimodal.


```python
import seaborn as sns

sns.countplot(x = 'diagnosis', data = df)
plt.title('Diagnosis of Samples')
plt.xlabel(xlabel = None)
plt.ylabel('Counts')
plt.show()
```

![png](/images/breastcancer/output_22_0.png)


Our target variable has approximately 150 more benign samples than malignant samples.  While it is expected that benign values would likely outnumber malignant values, this means that accuracy as a target metric for our machine learning algorithms will be deceptively high.  We will focus on the F1, precision, and recall scores instead.

While these values are skewed which could affect the model, the outliers could also be possibly correlated with more malignant disease.  I will not transform these skewed values to be as accurate as possible.

Using swarm plots, the numeric variables were compared with the target diagnosis of benign or malignant.

![png](/images/breastcancer/output_25_0.png)



![png](/images/breastcancer/output_25_1.png)



![png](/images/breastcancer/output_25_2.png)



![png](/images/breastcancer/output_25_3.png)

In general, the mean values for many of the predictor variables tended to be higher with malignant diagnoses versus benign diagnoses.

![png](/images/breastcancer/output_25_11.png)


![png](/images/breastcancer/output_25_14.png)

The standard error values for the predictors tended to be equivalent between the malignant and benign diagnoses.

![png](/images/breastcancer/output_25_16.png)


![png](/images/breastcancer/output_25_18.png)


![png](/images/breastcancer/output_25_19.png)


![png](/images/breastcancer/output_25_20.png)

The malignant tumors seemed to have higher worst texture, compactness, concavity, and concave points.

## Machine Learning Fitting and Prediction

In general, the goal of these machine learning algorithms wouuld be to eliminate false negatives as much as possible.  For example, we would not want our models to predict a benign diagnosis when a cancer was present.  This would be devastating.  While false positives (predicting cancer when it is benign) would be quite distressing, it would not be fatal as a missed cancer diagnosis would be.  The algorithms will be judged on this ability as well as Precision, Recall, and F1 scores.

### Logistic Regression


```python
#Running Logistic Regression Method to Predict Malignant or Benign

#Importing Packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#Creating Features and Target Objects
features = df.loc[:, df.columns != 'diagnosis']
target = df['diagnosis']

#Creating Standardizer
standardizer = StandardScaler()

#Creating Logistic Regression Object
logit = LogisticRegression()

#Standardizing Features
features_standardized = standardizer.fit_transform(features)

#Train Test 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state =1)

#Fitting Data to Logistic Regression Classifier
logreg = logit.fit(features_train, target_train)

#Generating Confusion Matrix/Classification Report
target_pred = logit.predict(features_test)
test0 = np.array(target_test)
predictions0 = np.array(target_pred)
print("Confusion Matrix:\n", confusion_matrix(test0, predictions0),'\n')
print("Classification Report:\n", classification_report(test0, predictions0))
```

    Confusion Matrix:
     [[71  1]
     [ 2 40]]

    Classification Report:
                   precision    recall  f1-score   support

               B       0.97      0.99      0.98        72
               M       0.98      0.95      0.96        42

        accuracy                           0.97       114
       macro avg       0.97      0.97      0.97       114
    weighted avg       0.97      0.97      0.97       114



This model performed quite well.  There were 2 misclassifications of benign lesions that were actually malignant.  F1 score for benign lesions was 0.98 and malignant lesions was 0.96 which are quite high.  Next we will check the feature importance of the different factors in the model based on their coefficients.


```python
#Checking Feature Importance Based on Coefficients

#Importing Packages
from yellowbrick.model_selection import FeatureImportances

#Getting Labels and Checking Feature Importances For Logistic Regression Model
labels = list(map(lambda x: x.title(), features))
viz = FeatureImportances(logreg, labels=labels)
viz.fit(features_train, features_test)
viz.show()
```


![png](/images/breastcancer/output_31_0.png)


Looking at the relative feature importance using the logistic regression model, radius_se, radius_mean, concavepoints_mean, texture/concavepoints/symmetry worst and concavity_mean/worst all had fairly high relative importances based on their coefficients.

It is not surprising that the radius mean and standard error values were highly important in the model since cell size is proportional to malignant cells (generally).  It is also interesting that there were a lot of variables with worst values that were highly important.  This could suggest that the values that are the most abnormal could be associated with predicting malignant or benign lesions.

### K Nearest Neighbors

First the data was fit on the KNN model and then the hyperparameters were optimized.

```python
#Running KNN To Predict Malignant or Benign

#Using Same Features and Target as Previous Example though Will Need to Get Dummy Variables for Target
target = pd.get_dummies(df['diagnosis'])

#Importing Packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#Creating Standardizer
standardizer = StandardScaler()

#Standardizing Features
features_standardized = standardizer.fit_transform(features)

#Train/Teset 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state = 1)

#Creating KNN Object With K of 3 Before Hyperparameter Tuning
knn = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

#Fitting Classifier on Training Data
knn.fit(features_train, target_train)

#Creating Confusion Matrix/Classification Report
target_pred = knn.predict(features_test)
test = np.array(target_test).argmax(axis=1)
predictions = np.array(target_pred).argmax(axis=1)
print("Confusion Matrix:\n", confusion_matrix(test, predictions),'\n')
print("Classification Report:\n", classification_report(test, predictions))
```

    Confusion Matrix:
     [[69  3]
     [ 7 35]]

    Classification Report:
                   precision    recall  f1-score   support

               0       0.91      0.96      0.93        72
               1       0.92      0.83      0.88        42

        accuracy                           0.91       114
       macro avg       0.91      0.90      0.90       114
    weighted avg       0.91      0.91      0.91       114


Using the following hyperparameters, the KNN model was re-run.

    Best Leaf Size: 1
    Best p: 1
    Best n_neighbors: 7
    Best Metric: minkowski
    Best Weights: uniform


    Confusion Matrix:
     [[70  2]
     [ 7 35]]

    Classification Report:
                   precision    recall  f1-score   support

               0       0.91      0.97      0.94        72
               1       0.95      0.83      0.89        42

        accuracy                           0.92       114
       macro avg       0.93      0.90      0.91       114
    weighted avg       0.92      0.92      0.92       114



Feature importance was not done on the KNN model since it is not directly applicable to this algorithm.  However, this model performed worse than the logistic regression model with F1-scores of 0.94 for benign lesions and 0.89 for malignant lesions.  There were also 7 false negatives, which is much higher than the Logistic Regression model.

### Random Forest Classifier


```python
#Running Random Forest Algorithm on Data

#Importing Packages
from sklearn.ensemble import RandomForestClassifier

#Resetting Features and Targets
features = df.loc[:, df.columns != 'diagnosis']
target = df['diagnosis']

#Creating Standardizer
standardizer = StandardScaler()

#Standardizing Features
features_standardized = standardizer.fit_transform(features)

#Train/Teset 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state = 1)

#Creating Random Forest Object
rf = RandomForestClassifier(n_estimators= 100, random_state = 1)

#Fitting Classifier on Training Data
rf.fit(features_train, target_train)

#Creating Confusion Matrix/Classification Report
target_pred = rf.predict(features_test)
test = np.array(target_test)
predictions = np.array(target_pred)
print("Confusion Matrix:\n", confusion_matrix(test, predictions),'\n')
print("Classification Report:\n", classification_report(test, predictions))
```

    Confusion Matrix:
     [[72  0]
     [ 5 37]]

    Classification Report:
                   precision    recall  f1-score   support

               B       0.94      1.00      0.97        72
               M       1.00      0.88      0.94        42

        accuracy                           0.96       114
       macro avg       0.97      0.94      0.95       114
    weighted avg       0.96      0.96      0.96       114



The random forest model performed slightly better than the KNN model with an F1 score of 0.97 for benign lesions and an F1 score of 0.94 for malignant lesions.  Of more concern is that 5 benign lesions were classified as benign when they were actually malignant.

Next, for comparison between this and the logistic regression model, we'll rank feature importance in this model as well.


```python
#Ranking Feature Importance Based on the Random Forest Algorithm

labels1 = list(map(lambda x: x.title(), features))
viz = FeatureImportances(rf, labels=labels)
viz.fit(features_train, features_test)
viz.show()
```


![png](/images/breastcancer/output_41_0.png)

The results of this were quite interesting as well.  Concavepoints_worst, concavepoints_mean, radius_mean, concavity_mean, and concavity_worst were all ranked highly with relative importance.  These are very similar to the logistic regression model.

This would seem to suggest that radius_mean and concavepoints_worst/mean all seem to be important whichever model you are using and perhaps could be a focus of future projects.

Overall, the Logistic Regression model seems to be the best option of the three models tested.  Due to the imbalanced target class, accuracy alone is not a good measure of performance since in an unbalanced class, the algorithm by chance could be more likely to guess a certain outcome because it knows that that is the most likely answer.

The F1 scores for the Logistic Regression model were the highest of the 3 models tested.  This model also minimized the number of false negatives or missed cancer diagnoses.

Limitations of this dataset include older data as well as limited data points.  Further projects would require thousands, perhaps millions of more current data points.  Healthcare demographics as well as cancer diagnosis and treatment standards of care change rapidly so current data is paramount.

This project demonstrates that machine learning could be potentially useful as an adjunct to standard patient care.  The goal should be to use machine learning models as an adjunct to flag potentially high-risk findings that should be further investigated before disregarding.  This could be something as simple as the model indicating that there are several highly suspicious features of malignancy and have a physician review for final diagnosis to either concur or dispute that result.  Outcome improvement is paramount; a healthcare system must strive to deliver the best quality care as possible.  This is a basic tenet of treating patients and this project suggests that machine learning could be beneficial.

__For full code of the project, please refer to my GitHub repository under Applied Data Science.__
