<!-- HEADER SECTION -->

<div class='header'> 
<!-- Your header image here -->
<div class='headingImage' id='mainHeaderImage' align="center">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/NYC2.jpg" width='1200' height='500' ></img>
</div>

<!-- Put your badges here, either for fun or for information -->
<div align="center">
    <!-- Project Type -->
    <img src="https://img.shields.io/badge/Project Type-Machine Learning-purple?style=flat-square">
    <!-- Maintained? -->
    <img src="https://img.shields.io/badge/Maintained%3F-IN PROG-blue?style=flat-square"></img>
    <!-- License? (MIT is Standard, make sure you license your project via github) -->
    <img src="https://img.shields.io/github/license/boogiedev/automotive-eda?style=flat-square">
    <!-- Commit Activity? (Fill in the blanks) -->
    <img src="https://img.shields.io/github/commit-activity/m/your_username/your_repo_name?style=flat-square">
</div>

</br>

<!-- Brief Indented Explaination, you can choose what type you want -->
<!-- Type 1 -->
>NYC Airbnb Rental Prices Prediction



<!-- TABLE OF CONTENTS SECTION -->
<!-- 
In page linkings are kind of weird and follow a specific format, it can be done in both markdown or HTML but I am sticking to markdown for this one as it is more readable. 

Example:
- [Title of Section](#title-of-section)
  - [Title of Nested Section](#title-of-nested-section)

## Title of Section

### Title of Nested Section

When linking section titles with spaces in between, you must use a '-' (dash) to indicate a space, and the reference link in parentheses must be lowercase. Formatting the actual title itself has to be in markdown as well. I suggest using two hashtags '##' to emphasize it is a section, leaving the largest heading (single #) for the project title. With nested titles, just keep going down in heading size (###, ####, ...)
-->

## Table of Contents

<!-- Overview Section -->

- [Overview](#overview)
  - [Background & Motivation](#context)
  - [Goal](#context)

<!-- Section 1 -->
- [EDA](#context)
    - [Raw data](#visualizations)
    - [Data Analysis](#context)

<!-- Section 2 -->
- [Feature engineering](#visualizations)

<!-- Section 3 -->
- [Create model](#visualizations)
    - [Choosing model](#visualizations)
    - [Feature engineering agian](#visualizations)
    - [Try best hyperparameters](#visualizations)
    - [create final model](#visualizations)
    - [Evaluate model](#visualizations)

<!-- Section 4 -->
- [Interpret model](#context)

<!-- Section 5 -->
- [Future Steps](#context)




<!-- Optional Line -->
---



## Overview

### Background & Motivation

Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.

Source: kaggle.com
### Goal


<!-- SECTION 1 -->
## EDA
### Raw data
<img src=''></img>

I search the data from kaggle.com. This dataset has around 49,000 observations in it with 16 columns and it is a mix text, categorical and numeric values.

Scan the data to determine what I need.

### Data Analysis
Get some intuitive sense of the relationships between numeric feature variables and Price
<img src='image/correlation_matrix_v0.png'></img>



<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/scatter_matrix.png?raw=true'></img>

Take a closer look at the data by navigate different features.

Navigate "neighbourhood_group" : NYC borough

Manhattan        21661
Brooklyn         20104
Queens            5666
Bronx             1091
Staten Island      373

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/map_of_neighbourhood_group.png?raw=true'></img>

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/map_of_neighbourhood_group.png?raw=true'></img>

Navigate "neighbourhood": NYC neighbourhood
<img src=''></img>

Navigate "room_type": type of listing
Entire home/apt    25409
Private room       22326
Shared room         1160

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/map_of_room_type.png?raw=true'></img>

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/mean_price_by_room_type.png?raw=true'></img>

navigate "minimum_nights": required minimum nights stay
<img src=''></img>

Navigate "name": listing name
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/name_wordcloud.png?raw=true'></img>


Closer look at "price" : listing price

df.price.describe()
count    48895.000000
mean       152.720687
std        240.154170
min          0.000000
25%         69.000000
50%        106.000000
75%        175.000000
max      10000.000000
Name: price, dtype: float64

```python
    (df['price'] > 2000).sum()
    86
    (df['price'] > 1000).sum()
    239
```
There are just less than 0.5% price is greater than $1,000











<!-- SECTION 2 -->
## Feature engineering


<img src=?raw=true' width='800' height='auto'></img>




<!-- SECTION 3 -->
## Create model
#### Choosing model
1)

#### Try best hyperparameters
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/rf_MSE_vs_Num_Estimators.png?raw=true' width='800' height='auto'></img>
#### create final model

#### Evaluate model





<!-- SECTION 4 -->

#### Interpret model

Feature Importances

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/top_10_feature_importances.png?raw=true' width='800' height='auto'></img>








<img src='?raw=true'></img>


<!-- SECTION 5 -->
## Future Steps
Next, I want to deep-dive the data.







<!-- Another line -->
---
