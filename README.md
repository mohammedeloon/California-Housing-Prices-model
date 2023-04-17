# California-Housing-Prices-model
A prediction of a district’s median housing price
## Frame the Big Picture
This model learns from the dataset provided and predicts the median housing price of each block group in California, based on many other features.
The output of this ML model will be fed to another ML system with other signals. Finally, the downstream system will decide whether it is
worth investing in a given area.
Note: getting this right is critical as it directly affects the revenue.

![image](https://user-images.githubusercontent.com/102263730/232287745-385ac349-a722-466c-8bb2-36cf9a711635.png)
## Quick EDA
![image](images/skim.png)
![image](images/plot.png)
## Create a Test Set 
In this stage, I created two sets: one for training(80%) and another for test(20%) using stratified  sampling based on the income 
category. The income category is a feature built on top of median income value feature.
```python
df['income_cat'] = pd.cut(df['median_income'] , bins=[0. , 1.5 , 3.0 , 4.5 , 6.0 , np.inf]  , labels=[1,2,3,4,5])
df['income_cat'].hist()
```

![image](images/myplot.png)

This way we can ensure that the test set is representative of the various categories of incomes in
the whole dataset.

## Visualize the Data to Gain Insights
![image](images/geographical_scatterplot.png)
![image](images/geographical_scatterplot_with_alpha.png)
![image](images/housing_price_scatter.png)
![image](images/scatter_matrix.png)
![image](images/scatter_house_value_house_income.png)
