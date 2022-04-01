![Real Estate Image](https://www.pexels.com/photo/house-lights-turned-on-106399/)
# Phase Two Project

For our phase two project we were tasked with creating a regression model for a real estate stakeholder.

We decided our stakeholder would be a man or woman looking for a house that will appreciate in value and is close to a middle school for their children.


# Data Understanding

* King County real estate data for homes sold in and around King County, Washington.
* Middle school locations in King County. We are able to calculate the distances from the houses in King County real estate data.


'''

df = pd.read_csv('data/kc_house_data.csv')
df2 = pd.read_csv('data/middle_school_hd.csv')
df.head()
'''
# Data Preperation
