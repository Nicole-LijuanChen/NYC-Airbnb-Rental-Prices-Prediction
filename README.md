<!-- HEADER SECTION -->

<div class='header'> 
<!-- Your header image here -->
<div class='headingImage' id='mainHeaderImage' align="center">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/NYC2.jpg" width='1200' height='500' ></img>
</div>

<!-- Put your badges here, either for fun or for information -->

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
    - [Data understanding](#context)

<!-- Section 2 -->
- [Feature engineering](#visualizations)

<!-- Section 3 -->
- [Modeling](#visualizations)
    - [Choosing model](#visualizations)
    - [Feature engineering agian](#visualizations)
    - [Try best hyperparameters](#visualizations)
    - [create final model](#visualizations)
    - [Evaluate model](#visualizations)

<!-- Section 4 -->
- [Bussiness insights](#context)

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
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/raw_data.png'></img>

This dataset has around 49,000 observations in it with 16 columns and it is a mix text, categorical and numeric values.

Scan the data infomation:
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/data_info.png'></img>

### Data understanding through visualizations
#### Get some intuitive sense of the relationships between numeric feature variables and Price

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/scatter_matrix.png?raw=true'></img>

According the scatter matrix, there is not a strong linear relationship between the numeric features with price.

#### Take a closer look at the data by navigate different features.

#### Navigate "neighbourhood_group" : NYC borough

Count frequency: 

Manhattan        21661;    
Brooklyn         20104;  
Queens            5666;  
Bronx             1091; 
Staten Island      373

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/map_of_neighbourhood_group.png'></img>

Mean price by neighbourhood_group              
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/mean_price_by_neighbourhood_group.png?raw=ture'></img>

#### Navigate "neighbourhood": NYC neighbourhood
<center class="half">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/nbh_value_counts.png" width="400"/><img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/nbh_counts.png" width="400"/>
</center>

#### Navigate "room_type": type of listing

Count frequency: 

Entire home/apt    25409; 
Private room       22326; 
Shared room         1160

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/map_of_room_type.png?raw=true'></img>

Mean price by neighbourhood_group  

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/mean_price_by_room_type.png?raw=true'></img>

#### Navigate "minimum_nights": required minimum nights stay
<center class="half">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/minimum_nights_distribution.png" width=300/>
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/min_nights_describe.png" width="300"/>
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/min_nights_counts.png" width="300"/>
</center>

#### Navigate "name": listing name

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/name_wordcloud.png?raw=true'></img>


#### Closer look at "price" : listing price

<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/price_distribution.png" width='800' height='auto'/>

<center class="half">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/price_describe.png" width="400"/>
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/price_counts.png" width="400"/>
</center>

#### There are just less than 0.5% listing's price is greater than $1,000











<!-- SECTION 2 -->
## Feature engineering
#### - Fill Nan values using the specified method.
        According to data.info, there are 4 cloumns missing some values.
        lname,host_name,last_review and reviews_per_month
    
#### - Convert categorical variable into numeric variables(0/1).
        According to data understanding, there are 3 important categorical features:
        neighbourhood_group,neighbourhood,room_type

#### - Drop some columns that have a low correlation with "price".
     id, host_id, host_name ....

The processed data:

<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/processed_data.png" width='800' height='auto'></img>




<!-- SECTION 3 -->
## Modeling
#### Define models
- Try 4 regressor models

<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/define_models.png" width='800' height='auto'></img>   


- Evaluate models

<img src=https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/evaluate_models_v0.png width='800' height='auto'></img>

#### Feature engineering again
- Drop outliers
    - price:
        The mean of price is 152.72, but the max price is 10000 that is not a reasonal price for me. 
        There are just less than 0.5% listing's price is greater than $1,000.
        #### Drop the rows whose price is greater than 1000 and equal to 0

    - minimum_nights:
        The minimum night stay policy on Airbnb is the minimum number of nights that a guest can book a short-term vacation rental. 
        Short-term stays means less than 30 nights at a time.
        #### Replace the df['minimum_nights'] >30 with 30
- Drop low correlation columns
    -latitude
    -longitude
- Convert text variable into numeric variables
     name -> nema_length
- Look at correlations again
####    Before  VS  After
<div align="center">
    <img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/correlation_matrix_v0.png" width="500"/><img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/correlation_matrix_v1.png" width="500"/>
<div>


- Evaluate models again

The models performaced have improved!!

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/evaluate_models_v1.png' width='800' height='auto'></img>

#### Try best hyperparameters


<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/rf_MSE_vs_Num_Estimators.png" width="600"/>
<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/rf_MSE_vs_Num_Trees.png" width="600"/>
<img src="https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/rf_MSE_vs_Num_Features.png" width="600"/>


#### create final model: Random Forest Regressor

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/final_model.png' width='800' height='auto'></img>

#### Evaluate model

<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/evaluate_final.png' width='800' height='auto'></img>


<!-- SECTION 4 -->

#### Bussiness insights

Feature Importances
<img src='https://github.com/Nicole-LijuanChen/NYC-Airbnb-Rental-Prices-Prediction/blob/master/images/top_10_feature_importances.png?raw=true' width='800' height='auto'></img>








<!-- SECTION 5 -->
## Future Steps
Next, I would like to search some related dataset, such as: the ratings, the reviews, numbers of booking. And then analyze the key fetures for customers to choose a listing.







<!-- Another line -->
---
