# SDSS-Datathon-2025

Our project submission for the first SDSS Datathon held on the University of Toronto's St. George campus in 2025.
The Datathon was held over 2 days, with a main goal of analysing datasets related to either Toronto's transit, and/or its real estate market.

Our group of 4 chose to analyse its real estate market, focusing mainly on a general solution to predict housing prices given several variables, such as the number of certain rooms, management fee costs, total size, etc.

We tried 3 different approaches to a general solution. A decision-tree-based algorithm, a linear regression model, and a nonlinear model. We trained each model on 80% of the dataset, using the remaining 20% as a validation set. The decision-tree algorithm was our best model based on the average difference between predicted and actual data. The linear regression model came closely behind, but the nonlinear fit being by the worst by being over twice as inaccurate as the other two.

We tried a few ways to augment our provided dataset by trying to match location data of each estate with publicly available crime rate, nearest TTC station, and other data, but ultimately decided against it given the time constraint.

We made a key insight into the nature of the pricing, which was that only two select variables had a significant impact on the overall price, namely the overall size of the listing and its corresponding management cost. The other variables had no significant impact on the pricing, at least by themselves. As a result, our best models only factored in 2 or 3 independent variables into the predictions.

AI tool usage: ChatGPT was used throughout to help with various bits of syntax and model suggestions, but key insights were all obtained through our own implementation of each model.

Video presentation link: https://youtu.be/rfTVIaOaru0