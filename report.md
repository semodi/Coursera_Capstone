# Finding (Pareto) Optimal Neighborhoods in NYC

## Introduction

Finding a place to live is never easy, this is especially true in a mega-city like New York. The options seem endless, and different trade-offs need to be considered when hunting for apartments: Do I want to live closer to my workplace but pay a higher rent, or do I possibly want to move to a more quiet, safer, residential neighborhood sacrificing valuable time during my commute?

When it comes to choosing the right place to live, every individual will have different priorities, so there is no "one size fits all" solution. In terms of formal decision-making theory, this problem can be cast as a multi-objective or Pareto optimization.
As the name suggests, rather than optimizing a variable (in this case the neighborhood) to minimize a single cost-function (i.e. rent) we want to simultaneously optimize several cost-functions. While not entirely accurate in the mathematical sense we will refer to these cost-functions loosely as metrics in the remainder of this report and we will specify the metrics used in the following section.

The goal of this report will be to find Pareto efficient (PE) (aka Pareto optimal) NYC Neighborhoods. The easiest way to understand Pareto efficiency is in terms of a negative example: Let us assume we want to find a neighborhood that is both cheap and safe. The metrics are therefore median rent and crime rate. If a neighborhood is **not** Pareto efficient, we can always find a different neighborhood that improves at least one of the metrics while not impairing any other ones. Conversely, if this is not possible, the neighborhood is called Pareto efficient.

![pareto_example_cleaned.png](fig/pareto_example_cleaned.png)

***Figure 1.*** *Pareto efficiency for synthetic dataset. The point highlighted in green is not PE as both Cost 1 and Cost 2 can be decreased by moving along the path indicated by the arrow.*

The benefits of this approach are clear, as simply finding the "best" neighborhood would require us to specify the relative importance of rent and crime rate. This relative importance, however, is highly subjective and will differ from person to person. 
Pareto efficiency serves as an objective tool to help the apartment-seeker find their best fit. Based on their personal preferences, they can go through the list of PE neighborhoods and choose the one closest to their liking. It should be noted that, from an optimization standpoint, it does not make sense to pick a non-PE neighborhood.


## Data

The metrics and associated data used in this report are the following

1. **Safety**
    
    To calculate the average safety of a given neighborhood I will combine data on the number of arrests made and the number of shootings. Both datasets are availabe at [NYC Open Data](https://opendata.cityofnewyork.us/).
    
1. **Rent**

    To gauge rent prices in every neighborhood, I decided to analyze the median rent for a one bedroom apartment. The data was obtained from [StreetEasy](https://streeteasy.com/blog/data-dashboard/?agg=Total&metric=Inventory&type=Sales&bedrooms=Any%20Bedrooms&property=Any%20Property%20Type&minDate=2010-01-01&maxDate=2020-05-01&area=Flatiron,Brooklyn%20Heights)

1. **Venue Density**

    Using the Foursquare API, in particular the "explore" endpoint, one can estimate the venue density in a given neighborhood. The venue density will be defined as the number of venues returned by Foursquare in a 500m radius around the neighborhood center.
    
1. **Distance from Subway**

    Combining [location data](https://data.cityofnewyork.us/Transportation/Subway-Entrances/drex-xx56) on subway entrances with [NYC geodata](https://data.beta.nyc/dataset/pediacities-nyc-neighborhoods) one can determine the average distance to the closest subway entrance for each neighborhood.

1. **Distance from Midtown**

    Given the shorter commute time, it might be desirable for some people to live as close as possible to their workplace. Both, the Financial District, Midtown Manhattan are the centers of economic activity in New York. I have chosen the latter to calculate this distance metric. 
    
Datasets only cover the year of 2019. Due to the lack of data regarding Staten Island, I disregarded the bourough in the final analysis.
    
## Methods

### NYC Open Data queries

Where possible, data was obtained from [NYC Open Data](https://opendata.cityofnewyork.us/) through their SODA API. This API allows for filtering, querying and aggregating data using a syntax reminiscient of SQL. For example to obtain data on all shootings that occured in 2019 one would use the following request:

``https://data.cityofnewyork.us/resource/833y-fsy8.json?$where=occur_date between '2019-01-01T00:00:00.000' and '2019-12-31T00:00:00.000' limit 1000000``

### Safety

To calculate the safety metric we combined data on arrests and data on shootings.

|Type| Total number in 2019|
|-----------------|-------|
|Arrests| 214617|
|Shootings| 1716|

Along with the location where the incident occured, the arrests data contained information about the type of felony/misdemeanor that was observed.

![](fig/arrests_freq.png)

Incidents, both in the case of arrests and shootings, were assigned to neighborhoods by a nearest neighbor algorithm. This means that for every incident, it was determined which neighborhood center (contained in the [NYC geodata](https://data.beta.nyc/dataset/pediacities-nyc-neighborhoods)) was closest to the incident using euclidean distance. 
To get an accurate treatment of distances, the coordinates provided as longitude and latitude first had to be multiplied by appropriate prefactors as so:

$$ d(A,B) = \sqrt{0.52^2(A_{long} - B_{long})^2 + 0.69^2 (A_{lat} - B_{lat})^2 } $$

This linear approximation to distances on a sphere works well for the small angles we are interested in.

The incident counts for each neighborhood were expressed as multiples of the average amount of incidents in all of NYC. Data on arrests and shootings was then combined together. As the normalization occured before the datasets were combined, more weight was automatically given to shootings. 


{ % include crime.html}



### Rent     

Obtaining rent data was challenging as no complete, openly avalaible datasets were obtainable online. 
The following data from [StreetEasy](https://streeteasy.com/blog/data-dashboard/?agg=Total&metric=Inventory&type=Sales&bedrooms=Any%20Bedrooms&property=Any%20Property%20Type&minDate=2010-01-01&maxDate=2020-05-01&area=Flatiron,Brooklyn%20Heights) proved to be the completest, but many neighborhoods are still missing as indicated by the black areas:

![](fig/rent_incomplete.png)

I decided to fill in the gaps with a predictive model. At this point I also made the decision to disregard Staten Island in the final analysis as data was simply not sufficient to draw reliable conclusions regarding this borough. 

Several ideas come to mind when trying to build a predictive model for rent prices. An important factor that determines rent prices is certainly location. One can therefore expect that a k-nearest-neighbor regression model will be able to predict rent prices by using information about adjacent neighborhoods. 

![](fig/rent_from_neighbors.png)

The model clearly becomes less reliable in regions of high rent and performs reasonably well in the  1500 to 2500 dollar regime. 

Another approach is to predict rent prices from past sales prices. This is possible as we can assume that both are correlated. Moreover sales prices are available for many neighborhoods for which rent prices are not. 
The data was fitted using Ridge Regression with Polynomial Features up to order 2.
![](fig/rent_from_sales.png)

The $R^2$ metric indicates that this model performes better than the one using nearest neighbors. However, sales data is not available for every neighborhood, so the final model will

For some neighborhoods where rent prices for 2019 are missing, historic prices are still available. For these neighborhoods one can simply model the average increase in rent over the years to get an estimate of current rent prices. 

![](fig/rent_from_past.png)

Among the three models introduced above, only the KNN algorithm can be used for every neighborhood, as both historic data as well as sales data is incomplete.
However, it seems that valuable information is contained in those datasets, so we should make use of them wherever possible. 
The solution to this is to build an ensemble model that takes the predicted prices from the above models as input and outputs one combined price. As model, a Random Forest Regressor was used for its ability to handle missing data. 
To simulate missing historic and sales data a random mask was applied to both training and test input.

![](fig/rent_from_ensemble.png)

The model performs better than the nearest neighbor regressor by itself, and due to the way it was fitted, can be applied to neighborhoods where historic and sales data is either present, partially complete or completely missing.



{ % include rent.html}

### Venue Density

Venue density in a given neighborhood was estimated as the amount of venues returned by the Foursquare API in a 500m radius around the neighborhood center. 

{ % include venues.html}

### Distance from Subway

To calculate the average distance to the closest subway station for every neighborhood a regular grid across NYC was generated. For every grid point, the distance to the closest subway station was determined. Using a ray casting algorithm, it was determined whether a given grid point was inside the polygon spanned by the neighborhood borders. 

![](fig/subway_grid.png)

Averaging over grid points in every neighborhood the "Distance from Subway" metric was created. 

{ % include subway.html}

### Distance from Midtown

As a proxy for "Distance from Midtown", the distance between every (geometric) neighborhood center and Times Square was calulated.

## Results


For visualization purposes, one calculate the pareto frontier for a subset of the metrics introduced above. The following graphic shows neighborhoods that are  pareto optimal with respect to Rent, Crime rate and average Rent.

{ % include pareto_3d.html}

Using all five metrics metioned in the Data section, one can measure the relative amount of Pareto optimal neighborhoods by borough:   
![](fig/percentage_pareto.png)

The clear winners are Manhattan and the Bronx, followed by Brooklyn and Queens in last place. 
But what is every Boroughs strongest asset? Looking only at Pareto optimal neighborhoods, one can calculate average value for every metric. For visualization purposes, I have normalized every metric to lie between zero and one, with zero being "worst" and one meaning "best". 


{ % include pareto_radial.html}

Not surprisingly, Manhattan is a clear winner when it comes to venue density and distance to Midtown (which, of course, is in Manhattan). With its many residential neighborhoods in the East, Queens can be considered the safest borough (not everywhere!).
Anyone who lives in the Bronx can attest to its unbeatable prices. It is also, in spite of its reputation, surprisingly safe, tying with Brooklyn and being safer on average than Manhattan. 

## Discussion

