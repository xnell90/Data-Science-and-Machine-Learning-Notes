
# Introduction

Many American cities have communal bike sharing stations where you can rent bicycles by the hour or day. Washington, D.C. is one of these cities. The District collects detailed data on the number of bicycles people rent by the hour and day.

Hadi Fanaee-T at the University of Porto compiled this data into a CSV file, which you'll be working with in this project. The file contains 17380 rows, with each row representing the number of bike rentals for a single hour of a single day. You can download the data from the University of California, Irvine's website. For more description about the columns see http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

From this data, can we predict the total number of bike rentals (cnt) based on the given predictor variables, such as season, year, month etc? We will try to do this using three different models: Linear Regression, Decision Tree, and Random Forest.


```python
#Import necessary libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load bike rentals data into a pandas dataframe
bike_rentals = pd.read_csv("bike_rental_hour.csv")

bike_rentals.drop(columns = ['casual', 'registered', 'instant'], inplace = True)
bike_rentals.head(10)
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
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0000</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.24</td>
      <td>0.2576</td>
      <td>0.75</td>
      <td>0.0896</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.20</td>
      <td>0.2576</td>
      <td>0.86</td>
      <td>0.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.32</td>
      <td>0.3485</td>
      <td>0.76</td>
      <td>0.0000</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



For more information about the above columns, visit the following link: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

Let's plot a histogram of the total number of bike rentals and see what we will find.


```python
bike_rentals.hist(column = 'cnt')
plt.title('''Histogram of the Total Number of Bike Rentals (Casual and Registered)''')
plt.xlabel("Total Number of Bike Rentals")
plt.ylabel("Frequency")
plt.show()
```


![png](output_5_0.png)


This histogram simply indicates that most total number of bike rentals at a given time frame are under 100, and there are a few instances where the total number of bike rentals are over 600. Next, let's compute the correlation matrix and see if there is a pair of features that are correlated with one another.


```python
#Computing the correlation matrix for our data
corr_bike_rentals = bike_rentals.corr()

sub_columns = ['season', 'yr', 'mnth', 
               'holiday', 'weekday',
               'workingday', 'weathersit', 'hum', 
               'windspeed', 'cnt']
corr_bike_rentals[sub_columns].loc[sub_columns]

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
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>season</th>
      <td>1.000000</td>
      <td>-0.010742</td>
      <td>0.830386</td>
      <td>-0.009585</td>
      <td>-0.002335</td>
      <td>0.013743</td>
      <td>-0.014524</td>
      <td>0.150625</td>
      <td>-0.149773</td>
      <td>0.178056</td>
    </tr>
    <tr>
      <th>yr</th>
      <td>-0.010742</td>
      <td>1.000000</td>
      <td>-0.010473</td>
      <td>0.006692</td>
      <td>-0.004485</td>
      <td>-0.002196</td>
      <td>-0.019157</td>
      <td>-0.083546</td>
      <td>-0.008740</td>
      <td>0.250495</td>
    </tr>
    <tr>
      <th>mnth</th>
      <td>0.830386</td>
      <td>-0.010473</td>
      <td>1.000000</td>
      <td>0.018430</td>
      <td>0.010400</td>
      <td>-0.003477</td>
      <td>0.005400</td>
      <td>0.164411</td>
      <td>-0.135386</td>
      <td>0.120638</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>-0.009585</td>
      <td>0.006692</td>
      <td>0.018430</td>
      <td>1.000000</td>
      <td>-0.102088</td>
      <td>-0.252471</td>
      <td>-0.017036</td>
      <td>-0.010588</td>
      <td>0.003988</td>
      <td>-0.030927</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>-0.002335</td>
      <td>-0.004485</td>
      <td>0.010400</td>
      <td>-0.102088</td>
      <td>1.000000</td>
      <td>0.035955</td>
      <td>0.003311</td>
      <td>-0.037158</td>
      <td>0.011502</td>
      <td>0.026900</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>0.013743</td>
      <td>-0.002196</td>
      <td>-0.003477</td>
      <td>-0.252471</td>
      <td>0.035955</td>
      <td>1.000000</td>
      <td>0.044672</td>
      <td>0.015688</td>
      <td>-0.011830</td>
      <td>0.030284</td>
    </tr>
    <tr>
      <th>weathersit</th>
      <td>-0.014524</td>
      <td>-0.019157</td>
      <td>0.005400</td>
      <td>-0.017036</td>
      <td>0.003311</td>
      <td>0.044672</td>
      <td>1.000000</td>
      <td>0.418130</td>
      <td>0.026226</td>
      <td>-0.142426</td>
    </tr>
    <tr>
      <th>hum</th>
      <td>0.150625</td>
      <td>-0.083546</td>
      <td>0.164411</td>
      <td>-0.010588</td>
      <td>-0.037158</td>
      <td>0.015688</td>
      <td>0.418130</td>
      <td>1.000000</td>
      <td>-0.290105</td>
      <td>-0.322911</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>-0.149773</td>
      <td>-0.008740</td>
      <td>-0.135386</td>
      <td>0.003988</td>
      <td>0.011502</td>
      <td>-0.011830</td>
      <td>0.026226</td>
      <td>-0.290105</td>
      <td>1.000000</td>
      <td>0.093234</td>
    </tr>
    <tr>
      <th>cnt</th>
      <td>0.178056</td>
      <td>0.250495</td>
      <td>0.120638</td>
      <td>-0.030927</td>
      <td>0.026900</td>
      <td>0.030284</td>
      <td>-0.142426</td>
      <td>-0.322911</td>
      <td>0.093234</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, there are at least two features in the data that are pairwise correlated. For example, if you look at season and month, they both have an associated correlation of 0.83. This is important because if we apply a linear regression model based on the above features, it will not perform well simply because linear regression models perfrom best if any pairwise features are not correlated with one another, i.e their correlation is close to 0.

# Feature Engineering

Before we can fit a model to our data, it is essential to engineer some features in our data. For now let's add a time label column.


```python
# time_label simply determines whether the hr number
# represents morning (1), afternoon (2), evening (3), 
# and night (4)
def assign_label(hr):
    if 6 <= hr and hr <= 12:
        return 1
    elif 12 <= hr and hr <= 18:
        return 2
    elif 18 <= hr and hr <= 24:
        return 3
    else:
        return 4

bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)
bike_rentals.head(10)
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
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
      <th>time_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0000</td>
      <td>16</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>40</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>32</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>13</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.24</td>
      <td>0.2576</td>
      <td>0.75</td>
      <td>0.0896</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0000</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.20</td>
      <td>0.2576</td>
      <td>0.86</td>
      <td>0.0000</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0000</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.32</td>
      <td>0.3485</td>
      <td>0.76</td>
      <td>0.0000</td>
      <td>14</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Import SKLearn Libraries, Train Test Split


```python
# Use mean absolute error from sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Split the data so that 80% of our data is our train set,
# while the rest is our test set.
train = bike_rentals.sample(frac = 0.8)
train.head(5)
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
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
      <th>time_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6293</th>
      <td>2011-09-24</td>
      <td>4</td>
      <td>0</td>
      <td>9</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.62</td>
      <td>0.5606</td>
      <td>0.88</td>
      <td>0.0000</td>
      <td>308</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14492</th>
      <td>2012-09-01</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.72</td>
      <td>0.6970</td>
      <td>0.74</td>
      <td>0.1343</td>
      <td>79</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3842</th>
      <td>2011-06-14</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.60</td>
      <td>0.6212</td>
      <td>0.49</td>
      <td>0.1940</td>
      <td>31</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2984</th>
      <td>2011-05-09</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.44</td>
      <td>0.4394</td>
      <td>0.72</td>
      <td>0.2537</td>
      <td>89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7811</th>
      <td>2011-11-27</td>
      <td>4</td>
      <td>0</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.34</td>
      <td>0.3636</td>
      <td>0.81</td>
      <td>0.0000</td>
      <td>31</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Preview our test set
test = bike_rentals[~bike_rentals.index.isin(train.index)]
test.head(5)
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
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>cnt</th>
      <th>time_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.32</td>
      <td>0.3485</td>
      <td>0.76</td>
      <td>0.0000</td>
      <td>14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.42</td>
      <td>0.4242</td>
      <td>0.77</td>
      <td>0.2836</td>
      <td>84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.46</td>
      <td>0.4545</td>
      <td>0.72</td>
      <td>0.2836</td>
      <td>106</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.40</td>
      <td>0.4091</td>
      <td>0.94</td>
      <td>0.2239</td>
      <td>28</td>
      <td>3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.44</td>
      <td>0.4394</td>
      <td>0.94</td>
      <td>0.2537</td>
      <td>17</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Here are the predictor variables we will use to predict cnt.
predictors = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
              'workingday', 'weathersit', 'temp', 'atemp', 'hum', 
              'windspeed', 'time_label']
```

# Linear Regression


```python
linear_regression = LinearRegression()
linear_regression.fit(train[predictors], train['cnt'])
prediction = linear_regression.predict(test[predictors])
mean_absolute_error(test['cnt'], prediction)
```




    98.32210396137293



Using a linear regression model gave us a prediction on the total number of bike rentals that is off by approximately 98 bikes.


```python
plt.xlabel("Actual # of Bike Rentals")
plt.ylabel("Predicted # of Bike Rentals")
plt.title("Actual vs Predicted")
plt.scatter(test['cnt'], prediction, s = 0.5)
plt.plot(np.linspace(0, 1000), np.linspace(0, 1000), color = 'red')
plt.show()
```


![png](output_19_0.png)


The above graph indicates that we are on average overpredicting the total number of bike rentals. Ideally, we want our points close to the read line.

# Decision Tree


```python
decision_tree_regression = DecisionTreeRegressor()
decision_tree_regression.fit(train[predictors], train['cnt'])
prediction = decision_tree_regression.predict(test[predictors])
mean_absolute_error(test['cnt'], prediction)
```




    34.293153049482164



Using a Decision Tree model gave us a prediction on the total number of bike rentals that is off by approximately 34 bikes. This is significantly much better than using linear regression.


```python
plt.xlabel("Actual # of Bike Rentals")
plt.ylabel("Predicted # of Bike Rentals")
plt.title("Actual vs Predicted")
plt.scatter(test['cnt'], prediction, s = 0.5)
plt.plot(np.linspace(0, 1000), np.linspace(0, 1000), color = 'red')
plt.show()
```


![png](output_24_0.png)


# Random Forest


```python
random_forest_regression = RandomForestRegressor(n_estimators = 200)
random_forest_regression.fit(train[predictors], train['cnt'])
prediction = random_forest_regression.predict(test[predictors])
mean_absolute_error(test['cnt'], prediction)
```




    24.434726002109702



Using a Random Forest model gave us a prediction on the total number of bike rentals that is off by approximately 24 bikes. This performs slightly better than a single decision tree. Moreover, if you increase the number of decision trees in your random forest, you would get better results.


```python
plt.xlabel("Actual # of Bike Rentals")
plt.ylabel("Predicted # of Bike Rentals")
plt.title("Actual vs Predicted")
plt.scatter(test['cnt'], prediction, s = 0.5)
plt.plot(np.linspace(0, 1000), np.linspace(0, 1000), color = 'red')
plt.show()
```


![png](output_28_0.png)

