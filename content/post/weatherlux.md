---
title: "Time series decomposition over open weather data from Luxembourg"
description: "Getting the trend, seasonal and random components from open source weather data of Luxembourg"
author: "Assaad Moawad"
date: "2017-03-22T14:29:24+01:00"
categories: ["machine learning", "time series"]
tags: ["time series", "decomposition", "profiling", "trend", "polynomial", "R language", "data analytics", "Luxembourg", "open data"]
image: "images/headers/thunders2.jpg"
---


Time series decomposition is a powerful statistical method that decomposes a signal into several components (usually a trend, a periodic and a random component).
These components can be used to do forecasting, prediction or extrapolation of missing data.

This topic is relatively old, since the [main research paper](http://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf) in the field dates back to 1990.
In R language, one of the most famous function for time series decomposition is the [stl function](https://www.r-bloggers.com/time-series-decomposition/).
More recently, Facebook has released an open source framework written in python called [prophet](https://research.fb.com/prophet-forecasting-at-scale/) for time series decomposition that is a little bit more advanced than the stl function, since it can take into account anomalies caused by non-periodic rare events.

At [datathings](www.datathings.com), we are experimenting with a different approach of time series decomposition. Since we aim at enabling live data analytics, we developed our own technology that uses online and live machine learning techniques to decompose time series on the fly. For the trend component we use a live polynomial learning algorithm, and for the periodic component we use a Gaussian Mixture Model (GMM) live profiling. The random component is what is left from signal after we remove the trend and the periodic component.

In order to see the results of the time series decomposition, we will use the open weather dataset of Luxembourg provided by [Open Data Luxembourg](https://data.public.lu/en/). The dataset consists of the historical monthly average of temperature records in Luxembourg since January 1947 and can be downloaded for free from [here](https://data.public.lu/en/datasets/monthly-meteorological-parameters-luxembourg-findel-airport-wmo-id-06590/). Here is the plot of the temperature data:

<iframe width="900" height="800" frameborder="0" scrolling="no" src="https://plot.ly/~datathings/10.embed"></iframe>

Due to earth periodic nature of the yearly rotation of earth around the sun, it is very hard to see from this graph the temperature trend over the years. Time series decomposition helps extracting the periodic component from the signal.
After passing it through a Gaussian profile, we can detect this periodic component. In January, the temperature average in Luxembourg  is around 0.5 °C, it raises to 17.5 °C in August, then falls down back to 0.5 °C in average for the January of the next year. Here is the result (we only show 2 years here for simplicity):

<iframe width="900" height="800" frameborder="0" scrolling="no" src="https://plot.ly/~datathings/11.embed"></iframe>

 After filtering out this periodic signal, we can then fit a polynomial curve of the temperature trend over the years.


<iframe width="900" height="800" frameborder="0" scrolling="no" src="https://plot.ly/~datathings/12.embed"></iframe>

As you can see from the graph, the temperature trend curve was stable till 1980s, then it increased by 2 degrees the last 36 years.
After removing the periodic and the trend signals from the original data, we are left with the random remainder. Here is what it looks like:

<iframe width="900" height="800" frameborder="0" scrolling="no" src="https://plot.ly/~datathings/13.embed"></iframe>

The interesting aspect about the random component is that it has an average value of 0. This is perfectly logical, since the time series decomposition goal is to move the average value of the signal to the periodic and linear trend.

In order to see the benefit of time series decomposition, we consider that our predictive model is the sum of the trend + periodic component. If we plot this predictive model (in orange) vs the real data in (blue) we get the following graph:
<iframe width="900" height="800" frameborder="0" scrolling="no" src="https://plot.ly/~datathings/14.embed"></iframe>


Finally, to validate our results, we run the time series decomposition stl function in R and using facebook Prophet, the results are displayed in the figures below.
They both confirm our result of increase in temperature of 2 degrees over the last 36 years. However both processes in R and with facebook prophet took more time to execute than our live approach and require to load the full dataset before learning. 

<center>Decomposition in R:</center>
![Decomposition in R](../../images/weatherlux/decomposition.png)

<center>Decomposition using Facebook Prophet:</center>
![Decomposition in R](../../images/weatherlux/facebook.png)
