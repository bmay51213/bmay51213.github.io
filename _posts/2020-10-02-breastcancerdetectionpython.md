---
title: "Detection of Breast Cancer of Biopsy Specimens Using Machine Learning"
date: 2020-10-02
classes: wide
header:
  image: "/images/breastcancer/1_yjsLGG-U9km84AvWLLmK8A.png"
excerpt: "Logistic Regression, KNN, Random Forest"
---

Breast cancer continues to be a significant culprit of morbidity and mortality, even today with our current medical advances.  The data that this dataset consists of was actually published in a medical journal article and data was collected at the University of Wisconsin (Wolberg, M.D. et al.¸1995). While this source is somewhat dated, my main goal with this project is to demonstrate that machine learning algorithms could be used to assist in cancer detection as adjunct to physician expertise.

__Dataset Description:__

The data is from the University of California - Irvine Machine Learning Repository and can be found here:  [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

This data was collected at the University of Wisconsin in 1995.  The file is in a csv format in which microscopic images of Fine Needle Aspirates (a type of biopsy) of suspicious breast tissue was digitized.  There are a total of 32 variables with 570 subjects.  They were examining suspicious masses in those without evidence of metastasis (distant spread of cancer to other parts of the body).


```python
#Loading Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Silencing Warnings
import warnings
warnings.filterwarnings('ignore')

#Importing Dataset and Viewing It
df = pd.read_csv('breast_cancer.csv')
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




```python
#Checking for Missing Values
df.isnull()
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>565</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>566</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>567</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>568</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 32 columns</p>
</div>



__Manually inspecting the dataframe reveals there to be no null values.__


```python
#Checking Datatypes to Ensure Coded Correctly
df.dtypes
```




    id                           int64
    diagnosis                   object
    radius_mean                float64
    texture_mean               float64
    perimeter_mean             float64
    area_mean                  float64
    smoothness_mean            float64
    compactness_mean           float64
    concavity_mean             float64
    concave points_mean        float64
    symmetry_mean              float64
    fractal_dimension_mean     float64
    radius_se                  float64
    texture_se                 float64
    perimeter_se               float64
    area_se                    float64
    smoothness_se              float64
    compactness_se             float64
    concavity_se               float64
    concave points_se          float64
    symmetry_se                float64
    fractal_dimension_se       float64
    radius_worst               float64
    texture_worst              float64
    perimeter_worst            float64
    area_worst                 float64
    smoothness_worst           float64
    compactness_worst          float64
    concavity_worst            float64
    concave points_worst       float64
    symmetry_worst             float64
    fractal_dimension_worst    float64
    dtype: object



__The target variable is categorical and is coded as M for malignant and B for benign.  There are no missing data in the CSV file.  There are case identifiers but other demographic information such as age, co-morbidities, and family history are not available.__

__The variables examined include measurements of digitized images on both cell size and shape.  These were already identified as either benign or malignant.  The data aims to determine if these specific characteristics can be used to predict whether a tumor was malignant or benign.__

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



```python
#Dropping the ID column
df = df.drop(columns = "id")

#Renaming Columns with Spaces in the Names

df = df.rename(columns={'concave points_mean': 'concavepoints_mean', 'concave points_se': 'concavepoints_se', 'concave points_worst': 'concavepoints_worst'})
```


```python
#Describing Dataset
print('Dataset Descriptive Statistics for Numerical Variables')
df.describe()
```

    Dataset Descriptive Statistics for Numerical Variables





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




```python
#Dataset Description for Categorical Variable
print('Dataset Descriptive Statistics for Categorical Variable')
df.describe(include=['O'])
```

    Dataset Descriptive Statistics for Categorical Variable





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




```python
#Performing Correlation Matrix to Look For High Levels of Correlation (>0.95)
corr_matrix = df.corr().abs()
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
      <th>radius_mean</th>
      <td>1.000000</td>
      <td>0.323782</td>
      <td>0.997855</td>
      <td>0.987357</td>
      <td>0.170581</td>
      <td>0.506124</td>
      <td>0.676764</td>
      <td>0.822529</td>
      <td>0.147741</td>
      <td>0.311631</td>
      <td>...</td>
      <td>0.969539</td>
      <td>0.297008</td>
      <td>0.965137</td>
      <td>0.941082</td>
      <td>0.119616</td>
      <td>0.413463</td>
      <td>0.526911</td>
      <td>0.744214</td>
      <td>0.163953</td>
      <td>0.007066</td>
    </tr>
    <tr>
      <th>texture_mean</th>
      <td>0.323782</td>
      <td>1.000000</td>
      <td>0.329533</td>
      <td>0.321086</td>
      <td>0.023389</td>
      <td>0.236702</td>
      <td>0.302418</td>
      <td>0.293464</td>
      <td>0.071401</td>
      <td>0.076437</td>
      <td>...</td>
      <td>0.352573</td>
      <td>0.912045</td>
      <td>0.358040</td>
      <td>0.343546</td>
      <td>0.077503</td>
      <td>0.277830</td>
      <td>0.301025</td>
      <td>0.295316</td>
      <td>0.105008</td>
      <td>0.119205</td>
    </tr>
    <tr>
      <th>perimeter_mean</th>
      <td>0.997855</td>
      <td>0.329533</td>
      <td>1.000000</td>
      <td>0.986507</td>
      <td>0.207278</td>
      <td>0.556936</td>
      <td>0.716136</td>
      <td>0.850977</td>
      <td>0.183027</td>
      <td>0.261477</td>
      <td>...</td>
      <td>0.969476</td>
      <td>0.303038</td>
      <td>0.970387</td>
      <td>0.941550</td>
      <td>0.150549</td>
      <td>0.455774</td>
      <td>0.563879</td>
      <td>0.771241</td>
      <td>0.189115</td>
      <td>0.051019</td>
    </tr>
    <tr>
      <th>area_mean</th>
      <td>0.987357</td>
      <td>0.321086</td>
      <td>0.986507</td>
      <td>1.000000</td>
      <td>0.177028</td>
      <td>0.498502</td>
      <td>0.685983</td>
      <td>0.823269</td>
      <td>0.151293</td>
      <td>0.283110</td>
      <td>...</td>
      <td>0.962746</td>
      <td>0.287489</td>
      <td>0.959120</td>
      <td>0.959213</td>
      <td>0.123523</td>
      <td>0.390410</td>
      <td>0.512606</td>
      <td>0.722017</td>
      <td>0.143570</td>
      <td>0.003738</td>
    </tr>
    <tr>
      <th>smoothness_mean</th>
      <td>0.170581</td>
      <td>0.023389</td>
      <td>0.207278</td>
      <td>0.177028</td>
      <td>1.000000</td>
      <td>0.659123</td>
      <td>0.521984</td>
      <td>0.553695</td>
      <td>0.557775</td>
      <td>0.584792</td>
      <td>...</td>
      <td>0.213120</td>
      <td>0.036072</td>
      <td>0.238853</td>
      <td>0.206718</td>
      <td>0.805324</td>
      <td>0.472468</td>
      <td>0.434926</td>
      <td>0.503053</td>
      <td>0.394309</td>
      <td>0.499316</td>
    </tr>
    <tr>
      <th>compactness_mean</th>
      <td>0.506124</td>
      <td>0.236702</td>
      <td>0.556936</td>
      <td>0.498502</td>
      <td>0.659123</td>
      <td>1.000000</td>
      <td>0.883121</td>
      <td>0.831135</td>
      <td>0.602641</td>
      <td>0.565369</td>
      <td>...</td>
      <td>0.535315</td>
      <td>0.248133</td>
      <td>0.590210</td>
      <td>0.509604</td>
      <td>0.565541</td>
      <td>0.865809</td>
      <td>0.816275</td>
      <td>0.815573</td>
      <td>0.510223</td>
      <td>0.687382</td>
    </tr>
    <tr>
      <th>concavity_mean</th>
      <td>0.676764</td>
      <td>0.302418</td>
      <td>0.716136</td>
      <td>0.685983</td>
      <td>0.521984</td>
      <td>0.883121</td>
      <td>1.000000</td>
      <td>0.921391</td>
      <td>0.500667</td>
      <td>0.336783</td>
      <td>...</td>
      <td>0.688236</td>
      <td>0.299879</td>
      <td>0.729565</td>
      <td>0.675987</td>
      <td>0.448822</td>
      <td>0.754968</td>
      <td>0.884103</td>
      <td>0.861323</td>
      <td>0.409464</td>
      <td>0.514930</td>
    </tr>
    <tr>
      <th>concavepoints_mean</th>
      <td>0.822529</td>
      <td>0.293464</td>
      <td>0.850977</td>
      <td>0.823269</td>
      <td>0.553695</td>
      <td>0.831135</td>
      <td>0.921391</td>
      <td>1.000000</td>
      <td>0.462497</td>
      <td>0.166917</td>
      <td>...</td>
      <td>0.830318</td>
      <td>0.292752</td>
      <td>0.855923</td>
      <td>0.809630</td>
      <td>0.452753</td>
      <td>0.667454</td>
      <td>0.752399</td>
      <td>0.910155</td>
      <td>0.375744</td>
      <td>0.368661</td>
    </tr>
    <tr>
      <th>symmetry_mean</th>
      <td>0.147741</td>
      <td>0.071401</td>
      <td>0.183027</td>
      <td>0.151293</td>
      <td>0.557775</td>
      <td>0.602641</td>
      <td>0.500667</td>
      <td>0.462497</td>
      <td>1.000000</td>
      <td>0.479921</td>
      <td>...</td>
      <td>0.185728</td>
      <td>0.090651</td>
      <td>0.219169</td>
      <td>0.177193</td>
      <td>0.426675</td>
      <td>0.473200</td>
      <td>0.433721</td>
      <td>0.430297</td>
      <td>0.699826</td>
      <td>0.438413</td>
    </tr>
    <tr>
      <th>fractal_dimension_mean</th>
      <td>0.311631</td>
      <td>0.076437</td>
      <td>0.261477</td>
      <td>0.283110</td>
      <td>0.584792</td>
      <td>0.565369</td>
      <td>0.336783</td>
      <td>0.166917</td>
      <td>0.479921</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.253691</td>
      <td>0.051269</td>
      <td>0.205151</td>
      <td>0.231854</td>
      <td>0.504942</td>
      <td>0.458798</td>
      <td>0.346234</td>
      <td>0.175325</td>
      <td>0.334019</td>
      <td>0.767297</td>
    </tr>
    <tr>
      <th>radius_se</th>
      <td>0.679090</td>
      <td>0.275869</td>
      <td>0.691765</td>
      <td>0.732562</td>
      <td>0.301467</td>
      <td>0.497473</td>
      <td>0.631925</td>
      <td>0.698050</td>
      <td>0.303379</td>
      <td>0.000111</td>
      <td>...</td>
      <td>0.715065</td>
      <td>0.194799</td>
      <td>0.719684</td>
      <td>0.751548</td>
      <td>0.141919</td>
      <td>0.287103</td>
      <td>0.380585</td>
      <td>0.531062</td>
      <td>0.094543</td>
      <td>0.049559</td>
    </tr>
    <tr>
      <th>texture_se</th>
      <td>0.097317</td>
      <td>0.386358</td>
      <td>0.086761</td>
      <td>0.066280</td>
      <td>0.068406</td>
      <td>0.046205</td>
      <td>0.076218</td>
      <td>0.021480</td>
      <td>0.128053</td>
      <td>0.164174</td>
      <td>...</td>
      <td>0.111690</td>
      <td>0.409003</td>
      <td>0.102242</td>
      <td>0.083195</td>
      <td>0.073658</td>
      <td>0.092439</td>
      <td>0.068956</td>
      <td>0.119638</td>
      <td>0.128215</td>
      <td>0.045655</td>
    </tr>
    <tr>
      <th>perimeter_se</th>
      <td>0.674172</td>
      <td>0.281673</td>
      <td>0.693135</td>
      <td>0.726628</td>
      <td>0.296092</td>
      <td>0.548905</td>
      <td>0.660391</td>
      <td>0.710650</td>
      <td>0.313893</td>
      <td>0.039830</td>
      <td>...</td>
      <td>0.697201</td>
      <td>0.200371</td>
      <td>0.721031</td>
      <td>0.730713</td>
      <td>0.130054</td>
      <td>0.341919</td>
      <td>0.418899</td>
      <td>0.554897</td>
      <td>0.109930</td>
      <td>0.085433</td>
    </tr>
    <tr>
      <th>area_se</th>
      <td>0.735864</td>
      <td>0.259845</td>
      <td>0.744983</td>
      <td>0.800086</td>
      <td>0.246552</td>
      <td>0.455653</td>
      <td>0.617427</td>
      <td>0.690299</td>
      <td>0.223970</td>
      <td>0.090170</td>
      <td>...</td>
      <td>0.757373</td>
      <td>0.196497</td>
      <td>0.761213</td>
      <td>0.811408</td>
      <td>0.125389</td>
      <td>0.283257</td>
      <td>0.385100</td>
      <td>0.538166</td>
      <td>0.074126</td>
      <td>0.017539</td>
    </tr>
    <tr>
      <th>smoothness_se</th>
      <td>0.222600</td>
      <td>0.006614</td>
      <td>0.202694</td>
      <td>0.166777</td>
      <td>0.332375</td>
      <td>0.135299</td>
      <td>0.098564</td>
      <td>0.027653</td>
      <td>0.187321</td>
      <td>0.401964</td>
      <td>...</td>
      <td>0.230691</td>
      <td>0.074743</td>
      <td>0.217304</td>
      <td>0.182195</td>
      <td>0.314457</td>
      <td>0.055558</td>
      <td>0.058298</td>
      <td>0.102007</td>
      <td>0.107342</td>
      <td>0.101480</td>
    </tr>
    <tr>
      <th>compactness_se</th>
      <td>0.206000</td>
      <td>0.191975</td>
      <td>0.250744</td>
      <td>0.212583</td>
      <td>0.318943</td>
      <td>0.738722</td>
      <td>0.670279</td>
      <td>0.490424</td>
      <td>0.421659</td>
      <td>0.559837</td>
      <td>...</td>
      <td>0.204607</td>
      <td>0.143003</td>
      <td>0.260516</td>
      <td>0.199371</td>
      <td>0.227394</td>
      <td>0.678780</td>
      <td>0.639147</td>
      <td>0.483208</td>
      <td>0.277878</td>
      <td>0.590973</td>
    </tr>
    <tr>
      <th>concavity_se</th>
      <td>0.194204</td>
      <td>0.143293</td>
      <td>0.228082</td>
      <td>0.207660</td>
      <td>0.248396</td>
      <td>0.570517</td>
      <td>0.691270</td>
      <td>0.439167</td>
      <td>0.342627</td>
      <td>0.446630</td>
      <td>...</td>
      <td>0.186904</td>
      <td>0.100241</td>
      <td>0.226680</td>
      <td>0.188353</td>
      <td>0.168481</td>
      <td>0.484858</td>
      <td>0.662564</td>
      <td>0.440472</td>
      <td>0.197788</td>
      <td>0.439329</td>
    </tr>
    <tr>
      <th>concavepoints_se</th>
      <td>0.376169</td>
      <td>0.163851</td>
      <td>0.407217</td>
      <td>0.372320</td>
      <td>0.380676</td>
      <td>0.642262</td>
      <td>0.683260</td>
      <td>0.615634</td>
      <td>0.393298</td>
      <td>0.341198</td>
      <td>...</td>
      <td>0.358127</td>
      <td>0.086741</td>
      <td>0.394999</td>
      <td>0.342271</td>
      <td>0.215351</td>
      <td>0.452888</td>
      <td>0.549592</td>
      <td>0.602450</td>
      <td>0.143116</td>
      <td>0.310655</td>
    </tr>
    <tr>
      <th>symmetry_se</th>
      <td>0.104321</td>
      <td>0.009127</td>
      <td>0.081629</td>
      <td>0.072497</td>
      <td>0.200774</td>
      <td>0.229977</td>
      <td>0.178009</td>
      <td>0.095351</td>
      <td>0.449137</td>
      <td>0.345007</td>
      <td>...</td>
      <td>0.128121</td>
      <td>0.077473</td>
      <td>0.103753</td>
      <td>0.110343</td>
      <td>0.012662</td>
      <td>0.060255</td>
      <td>0.037119</td>
      <td>0.030413</td>
      <td>0.389402</td>
      <td>0.078079</td>
    </tr>
    <tr>
      <th>fractal_dimension_se</th>
      <td>0.042641</td>
      <td>0.054458</td>
      <td>0.005523</td>
      <td>0.019887</td>
      <td>0.283607</td>
      <td>0.507318</td>
      <td>0.449301</td>
      <td>0.257584</td>
      <td>0.331786</td>
      <td>0.688132</td>
      <td>...</td>
      <td>0.037488</td>
      <td>0.003195</td>
      <td>0.001000</td>
      <td>0.022736</td>
      <td>0.170568</td>
      <td>0.390159</td>
      <td>0.379975</td>
      <td>0.215204</td>
      <td>0.111094</td>
      <td>0.591328</td>
    </tr>
    <tr>
      <th>radius_worst</th>
      <td>0.969539</td>
      <td>0.352573</td>
      <td>0.969476</td>
      <td>0.962746</td>
      <td>0.213120</td>
      <td>0.535315</td>
      <td>0.688236</td>
      <td>0.830318</td>
      <td>0.185728</td>
      <td>0.253691</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.359921</td>
      <td>0.993708</td>
      <td>0.984015</td>
      <td>0.216574</td>
      <td>0.475820</td>
      <td>0.573975</td>
      <td>0.787424</td>
      <td>0.243529</td>
      <td>0.093492</td>
    </tr>
    <tr>
      <th>texture_worst</th>
      <td>0.297008</td>
      <td>0.912045</td>
      <td>0.303038</td>
      <td>0.287489</td>
      <td>0.036072</td>
      <td>0.248133</td>
      <td>0.299879</td>
      <td>0.292752</td>
      <td>0.090651</td>
      <td>0.051269</td>
      <td>...</td>
      <td>0.359921</td>
      <td>1.000000</td>
      <td>0.365098</td>
      <td>0.345842</td>
      <td>0.225429</td>
      <td>0.360832</td>
      <td>0.368366</td>
      <td>0.359755</td>
      <td>0.233027</td>
      <td>0.219122</td>
    </tr>
    <tr>
      <th>perimeter_worst</th>
      <td>0.965137</td>
      <td>0.358040</td>
      <td>0.970387</td>
      <td>0.959120</td>
      <td>0.238853</td>
      <td>0.590210</td>
      <td>0.729565</td>
      <td>0.855923</td>
      <td>0.219169</td>
      <td>0.205151</td>
      <td>...</td>
      <td>0.993708</td>
      <td>0.365098</td>
      <td>1.000000</td>
      <td>0.977578</td>
      <td>0.236775</td>
      <td>0.529408</td>
      <td>0.618344</td>
      <td>0.816322</td>
      <td>0.269493</td>
      <td>0.138957</td>
    </tr>
    <tr>
      <th>area_worst</th>
      <td>0.941082</td>
      <td>0.343546</td>
      <td>0.941550</td>
      <td>0.959213</td>
      <td>0.206718</td>
      <td>0.509604</td>
      <td>0.675987</td>
      <td>0.809630</td>
      <td>0.177193</td>
      <td>0.231854</td>
      <td>...</td>
      <td>0.984015</td>
      <td>0.345842</td>
      <td>0.977578</td>
      <td>1.000000</td>
      <td>0.209145</td>
      <td>0.438296</td>
      <td>0.543331</td>
      <td>0.747419</td>
      <td>0.209146</td>
      <td>0.079647</td>
    </tr>
    <tr>
      <th>smoothness_worst</th>
      <td>0.119616</td>
      <td>0.077503</td>
      <td>0.150549</td>
      <td>0.123523</td>
      <td>0.805324</td>
      <td>0.565541</td>
      <td>0.448822</td>
      <td>0.452753</td>
      <td>0.426675</td>
      <td>0.504942</td>
      <td>...</td>
      <td>0.216574</td>
      <td>0.225429</td>
      <td>0.236775</td>
      <td>0.209145</td>
      <td>1.000000</td>
      <td>0.568187</td>
      <td>0.518523</td>
      <td>0.547691</td>
      <td>0.493838</td>
      <td>0.617624</td>
    </tr>
    <tr>
      <th>compactness_worst</th>
      <td>0.413463</td>
      <td>0.277830</td>
      <td>0.455774</td>
      <td>0.390410</td>
      <td>0.472468</td>
      <td>0.865809</td>
      <td>0.754968</td>
      <td>0.667454</td>
      <td>0.473200</td>
      <td>0.458798</td>
      <td>...</td>
      <td>0.475820</td>
      <td>0.360832</td>
      <td>0.529408</td>
      <td>0.438296</td>
      <td>0.568187</td>
      <td>1.000000</td>
      <td>0.892261</td>
      <td>0.801080</td>
      <td>0.614441</td>
      <td>0.810455</td>
    </tr>
    <tr>
      <th>concavity_worst</th>
      <td>0.526911</td>
      <td>0.301025</td>
      <td>0.563879</td>
      <td>0.512606</td>
      <td>0.434926</td>
      <td>0.816275</td>
      <td>0.884103</td>
      <td>0.752399</td>
      <td>0.433721</td>
      <td>0.346234</td>
      <td>...</td>
      <td>0.573975</td>
      <td>0.368366</td>
      <td>0.618344</td>
      <td>0.543331</td>
      <td>0.518523</td>
      <td>0.892261</td>
      <td>1.000000</td>
      <td>0.855434</td>
      <td>0.532520</td>
      <td>0.686511</td>
    </tr>
    <tr>
      <th>concavepoints_worst</th>
      <td>0.744214</td>
      <td>0.295316</td>
      <td>0.771241</td>
      <td>0.722017</td>
      <td>0.503053</td>
      <td>0.815573</td>
      <td>0.861323</td>
      <td>0.910155</td>
      <td>0.430297</td>
      <td>0.175325</td>
      <td>...</td>
      <td>0.787424</td>
      <td>0.359755</td>
      <td>0.816322</td>
      <td>0.747419</td>
      <td>0.547691</td>
      <td>0.801080</td>
      <td>0.855434</td>
      <td>1.000000</td>
      <td>0.502528</td>
      <td>0.511114</td>
    </tr>
    <tr>
      <th>symmetry_worst</th>
      <td>0.163953</td>
      <td>0.105008</td>
      <td>0.189115</td>
      <td>0.143570</td>
      <td>0.394309</td>
      <td>0.510223</td>
      <td>0.409464</td>
      <td>0.375744</td>
      <td>0.699826</td>
      <td>0.334019</td>
      <td>...</td>
      <td>0.243529</td>
      <td>0.233027</td>
      <td>0.269493</td>
      <td>0.209146</td>
      <td>0.493838</td>
      <td>0.614441</td>
      <td>0.532520</td>
      <td>0.502528</td>
      <td>1.000000</td>
      <td>0.537848</td>
    </tr>
    <tr>
      <th>fractal_dimension_worst</th>
      <td>0.007066</td>
      <td>0.119205</td>
      <td>0.051019</td>
      <td>0.003738</td>
      <td>0.499316</td>
      <td>0.687382</td>
      <td>0.514930</td>
      <td>0.368661</td>
      <td>0.438413</td>
      <td>0.767297</td>
      <td>...</td>
      <td>0.093492</td>
      <td>0.219122</td>
      <td>0.138957</td>
      <td>0.079647</td>
      <td>0.617624</td>
      <td>0.810455</td>
      <td>0.686511</td>
      <td>0.511114</td>
      <td>0.537848</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 30 columns</p>
</div>




```python
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




```python
#Visualizing Using Correlation Plot
plt.rcParams['figure.figsize'] = (20,20)

#Importing Seaborn
import seaborn as sns

sns.heatmap(corr_matrix)
plt.show()
```


![png](/images/breastcancer/output_12_0.png)


__Given the high rates of correlation of the above variables, we will drop them from the dataframe.  With a correlation >0.95, these are likely redundant and may introduce instability in our final algorithms.__


```python
#Dropping Highly Correlated Variables

df = df.drop(columns=['perimeter_mean', 'area_mean', 'perimeter_se', 'area_se', 'radius_worst', 'perimeter_worst', 'area_worst'])
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
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concavepoints_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>...</th>
      <th>concavepoints_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>texture_worst</th>
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
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.30010</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>...</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>17.33</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.08690</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>...</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>23.41</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.19740</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>...</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>25.53</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.24140</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>...</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>26.50</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.19800</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>...</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>16.67</td>
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
      <td>M</td>
      <td>21.56</td>
      <td>22.39</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>0.1726</td>
      <td>0.05623</td>
      <td>1.1760</td>
      <td>...</td>
      <td>0.02454</td>
      <td>0.01114</td>
      <td>0.004239</td>
      <td>26.40</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
    </tr>
    <tr>
      <th>565</th>
      <td>M</td>
      <td>20.13</td>
      <td>28.25</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>0.1752</td>
      <td>0.05533</td>
      <td>0.7655</td>
      <td>...</td>
      <td>0.01678</td>
      <td>0.01898</td>
      <td>0.002498</td>
      <td>38.25</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
    </tr>
    <tr>
      <th>566</th>
      <td>M</td>
      <td>16.60</td>
      <td>28.08</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>0.1590</td>
      <td>0.05648</td>
      <td>0.4564</td>
      <td>...</td>
      <td>0.01557</td>
      <td>0.01318</td>
      <td>0.003892</td>
      <td>34.12</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
    </tr>
    <tr>
      <th>567</th>
      <td>M</td>
      <td>20.60</td>
      <td>29.33</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>0.2397</td>
      <td>0.07016</td>
      <td>0.7260</td>
      <td>...</td>
      <td>0.01664</td>
      <td>0.02324</td>
      <td>0.006185</td>
      <td>39.42</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
    </tr>
    <tr>
      <th>568</th>
      <td>B</td>
      <td>7.76</td>
      <td>24.54</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1587</td>
      <td>0.05884</td>
      <td>0.3857</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.02676</td>
      <td>0.002783</td>
      <td>30.37</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 24 columns</p>
</div>




```python
#Finding Outliers Using Interquartile Range

#Loading Packages
from scipy import stats

#Calculating Interquartile Range
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3-q1
print(IQR)
```

    radius_mean                4.080000
    texture_mean               5.630000
    smoothness_mean            0.018930
    compactness_mean           0.065480
    concavity_mean             0.101140
    concavepoints_mean         0.053690
    symmetry_mean              0.033800
    fractal_dimension_mean     0.008420
    radius_se                  0.246500
    texture_se                 0.640100
    smoothness_se              0.002977
    compactness_se             0.019370
    concavity_se               0.026960
    concavepoints_se           0.007072
    symmetry_se                0.008320
    fractal_dimension_se       0.002310
    texture_worst              8.640000
    smoothness_worst           0.029400
    compactness_worst          0.191900
    concavity_worst            0.268400
    concavepoints_worst        0.096470
    symmetry_worst             0.067500
    fractal_dimension_worst    0.020620
    dtype: float64



```python
#Checking Which Values Fall Outside the Interquartile Range
(df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR))
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
      <th>compactness_mean</th>
      <th>compactness_se</th>
      <th>compactness_worst</th>
      <th>concavepoints_mean</th>
      <th>concavepoints_se</th>
      <th>concavepoints_worst</th>
      <th>concavity_mean</th>
      <th>concavity_se</th>
      <th>concavity_worst</th>
      <th>diagnosis</th>
      <th>...</th>
      <th>radius_se</th>
      <th>smoothness_mean</th>
      <th>smoothness_se</th>
      <th>smoothness_worst</th>
      <th>symmetry_mean</th>
      <th>symmetry_se</th>
      <th>symmetry_worst</th>
      <th>texture_mean</th>
      <th>texture_se</th>
      <th>texture_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>565</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>566</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>567</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>568</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 24 columns</p>
</div>



__There were several outliers identified using the interquartile range.  There were various categories where the outliers were identified with a wide variety of variables.  The most common variable where it was noted was compactness.  The majority of these outliers were mostly identified as malignant though there were some benign diagnoses with outliers in the compactness variables.__


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


    <Figure size 7200x7200 with 0 Axes>



![png](/images/breastcancer/output_18_1.png)


__Many of these variables are positively skewed.  The only variables that appear to be mostly normally distributed are symmetry_mean and possible fractal_dimension_mean.  Concave_points_worst actually appears to be bimodal.__

__Our target variable has approximately 150 more benign samples than malignant samples.  While it is expected that benign values would likely outnumber malignant values, this means that accuracy as a target metric for our machine learning algorithms will be deceptively high.  We will focus on the F1, precision, and recall scores instead.__

__While these values are skewed which could affect the model, the outliers could also be possibly correlated with more malignant disease.  I will not transform these skewed values to be as accurate as possible.__


```python
#Comparing Target Variable with our Explanatory Variables

#Setting Parameters
categories = df.diagnosis

fig,axs = plt.subplots(8,3, figsize=(30,30))
axs[0,0].scatter(categories, df.radius_mean)
axs[0,0].set_title('Benign/Malignant vs. Radius Mean Values')
axs[0,1].scatter(categories, df.texture_mean)
axs[0,1].set_title('Benign/Malignant vs. Texture Mean Values')
axs[0,2].scatter(categories, df.smoothness_mean)
axs[0,2].set_title('Benign/Malignant vs. Smoothness Mean Values')
axs[1,0].scatter(categories, df.compactness_mean)
axs[1,0].set_title('Benign/Malignant vs. Compactness Mean Values')
axs[1,1].scatter(categories, df.concavity_mean)
axs[1,1].set_title('Benign/Malignant vs. Concavity Mean Values')
axs[1,2].scatter(categories, df.concavepoints_mean)
axs[1,2].set_title('Benign/Malignant vs. Concave Points Mean Values')
axs[2,0].scatter(categories, df.symmetry_mean)
axs[2,0].set_title('Benign/Malignant vs. Symmetry Mean Values')
axs[2,1].scatter(categories, df.fractal_dimension_mean)
axs[2,1].set_title('Benign/Malignant vs. Fractal Dimension Mean Values')
axs[2,2].scatter(categories, df.radius_se)
axs[2,2].set_title('Benign/Malignant vs. Radius S.E. Values')
axs[3,0].scatter(categories, df.texture_se)
axs[3,0].set_title('Benign/Malignant vs. Texture S.E. Values')
axs[3,1].scatter(categories, df.smoothness_se)
axs[3,1].set_title('Benign/Malignant vs. Smoothness S.E. Values')
axs[3,2].scatter(categories, df.compactness_se)
axs[3,2].set_title('Benign/Malignant vs. Compactness S.E. Values')
axs[4,0].scatter(categories, df.concavity_se)
axs[4,0].set_title('Benign/Malignant vs. Concavity S.E. Values')
axs[4,1].scatter(categories, df.concavepoints_se)
axs[4,1].set_title('Benign/Malignant vs. Concave Points S.E. Values')
axs[4,2].scatter(categories, df.symmetry_se)
axs[4,2].set_title('Benign/Malignant vs. Symmetry S.E. Values')
axs[5,0].scatter(categories, df.fractal_dimension_se)
axs[5,0].set_title('Benign/Malignant vs. Fractal Dimension S.E. Values')
axs[5,1].scatter(categories, df.texture_worst)
axs[5,1].set_title('Benign/Malignant vs. Texture Worst Values')
axs[5,2].scatter(categories, df.smoothness_worst)
axs[5,2].set_title('Benign/Malignant vs. Smoothness Worst Values')
axs[6,0].scatter(categories, df.compactness_worst)
axs[6,0].set_title('Benign/Malignant vs. Compactness Worst Values')
axs[6,1].scatter(categories, df.concavity_worst)
axs[6,1].set_title('Benign/Malignant vs. Concavity Worst Values')
axs[6,2].scatter(categories, df.concavepoints_worst)
axs[6,2].set_title('Benign/Malignant vs. Concave Points Worst Values')
axs[7,0].scatter(categories, df.symmetry_worst)
axs[7,0].set_title('Benign/Malignant vs. Symmetry Worst Values')
axs[7,1].scatter(categories, df.fractal_dimension_worst)
axs[7,1].set_title('Benign/Malignant vs. Fractal Dimension Worst Values')
```




    Text(0.5, 1.0, 'Benign/Malignant vs. Fractal Dimension Worst Values')




![png](/images/breastcancer/output_21_1.png)


__Looking at the comparisons of our predictor variables vs. their diagnosis of malignant or benign, there are several interesting factors.  While there were some outliers, I still will keep these in the datasets for now.  In general, the malignant tumors seemed to have higher worst symmetry scores, worst concave points, worst concavity, worst compactness, and texture worst values.__

__In general, the radius mean tended to be larger with malignant tumors.__


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



__This model performed quite well.  There were 2 misclassifications of benign lesions that were actually malignant.  F1 score for benign lesions was 0.98 and malignant lesions was 0.96 which are quite high.  Next we will check the feature importance of the different factors in the model based on their coefficients.__


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


![png](/images/breastcancer/output_25_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x2219a37f400>



__Looking at the relative feature importance using the logistic regression model, radius_se, radius_mean, concavepoints_mean, texture/concavepoints/symmetry worst and concavity_mean/worst all had fairly high relative importances based on their coefficients.__

__It is not surprising that the radius mean and standard error values were highly important in the model since cell size is proportional to malignant cells (generally).  It is also interesting that there were a lot of variables with worst values that were highly important.  This could suggest that the values that are the most abnormal could be associated with predicting malignant or benign lesions.__


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




```python
#Hyperparameter Tuning for KNN Using GridSearch CV

#Creating Hyperparameter Grid
param_dist1 = {"leaf_size": list(range(1,50)),
              "n_neighbors": list(range(1,30)),
              "p": [1,2]}

#Creating New KNN Object
knn_2 = KNeighborsClassifier()

#Using GridSearch Object
clf = GridSearchCV(knn_2, param_dist1, cv=10, n_jobs = -1)

#Fitting Model
best_model = clf.fit(features_standardized, target)

print('Best Leaf Size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
print('Best Metric:', best_model.best_estimator_.get_params()['metric'])
print('Best Weights:', best_model.best_estimator_.get_params()['weights'])
```

    Best Leaf Size: 1
    Best p: 1
    Best n_neighbors: 7
    Best Metric: minkowski
    Best Weights: uniform



```python
#Rerunning KNN Using Our Tuned Parameters

#Creating Standardizer
standardizer = StandardScaler()

#Standardizing Features
features_standardized = standardizer.fit_transform(features)

#Train/Teset 80/20 Split
features_train, features_test, target_train, target_test = train_test_split(features_standardized, target, test_size = 0.2, random_state = 1)

#Creating KNN Object With K of 3 Before Hyperparameter Tuning
knn = KNeighborsClassifier(n_neighbors = 7, p = 1, leaf_size = 1, metric = "minkowski", weights = "uniform", n_jobs = -1)

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
     [[70  2]
     [ 7 35]]

    Classification Report:
                   precision    recall  f1-score   support

               0       0.91      0.97      0.94        72
               1       0.95      0.83      0.89        42

        accuracy                           0.92       114
       macro avg       0.93      0.90      0.91       114
    weighted avg       0.92      0.92      0.92       114



__Feature importance was not done on the KNN model since it is not directly applicable to this algorithm.  However, this model performed worse than the logistic regression model with F1-scores of 0.94 for benign lesions and 0.89 for malignant lesions.__


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



__The random forest model performed slightly better than the KNN model with an F1 score of 0.97 for benign lesions and an F1 score of 0.94 for malignant lesions.  Of more concern is that 5 benign lesions were classified as benign when they were actually malignant.__

__Next, for comparison between this and the logistic regression model, we'll rank feature importance in this model as well.__


```python
#Ranking Feature Importance Based on the Random Forest Algorithm

labels1 = list(map(lambda x: x.title(), features))
viz = FeatureImportances(rf, labels=labels)
viz.fit(features_train, features_test)
viz.show()
```


![png](/images/breastcancer/output_33_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x2219a28b970>



__The results of this were quite interesting as well.  Concavepoints_worst, concavepoints_mean, radius_mean, concavity_mean, and concavity_worst were all ranked highly with relative importance.  These are very similar to the logistic regression model.__

__This would seem to suggest that radius_mean and concavepoints_worst/mean all seem to be important whichever model you are using and perhaps could be a focus of future projects.__

__Overall, the Logistic Regression model seems to be the best option of the three models tested.  Due to the imbalanced target class, accuracy alone is not a good measure of performance since in an unbalanced class, the algorithm by chance could be more likely to guess a certain outcome because it knows that that is the most likely answer.__

__The F1 scores for the Logistic Regression model were the highest of the 3 models tested.  This model also minimized the number of false negatives or missed cancer diagnoses.__

__Limitations of this dataset include older data as well as limited data points.  Further projects would require thousands, perhaps millions of more current data points.  Healthcare demographics as well as cancer diagnosis and treatment standards of care change rapidly so current data is paramount.__

__This project demonstrates that machine learning could be potentially useful as an adjunct to standard patient care.  The goal is not to replace doctors; on the contrary, doctors are required to be on the forefront treating patients and are still experts in the field.  However, the goal should be to use machine learning models as an adjunct to flag potentially high-risk findings that should be further investigated before disregarding.  This could be something as simple as the model indicating that there are several highly suspicious features of malignancy and have a physician review for final diagnosis to either concur or dispute that result.  Outcome improvement is paramount; a healthcare system must strive to deliver the best quality care as possible.  This is a basic tenet of treating patients and this project suggests that machine learning could potentially help healthcare workers improve outcomes and make patient’s lives better.__


```python

```
