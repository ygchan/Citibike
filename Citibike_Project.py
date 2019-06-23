#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Citibike 2019 project, analyzing 2018 date of service trip data
# Author: George Chan, Sean Guo


# In[78]:


# Import library and modules
import pandas as pd
import glob
import numpy as np
import os
from datetime import timedelta
import matplotlib

# Future use
# https://markhneedham.com/blog/2018/09/18/matplotlib-remove-axis-legend/


# In[3]:


# Define where are the citibke 2018 zipfiles
data_folder = r'/Users/Study/Citibike_Data/'
print('The data are stored in: ' + data_folder)


# In[4]:


# Check what are the files in that folder
for folderName, subfolders, filenames in os.walk(data_folder):
    print('Here are the list of csv zip files you have downloaded.\n')
    print(folderName)
    
    number_of_csv_file = 0 # Total count of files read
    all_files = []         # List of file path string
    
    # Reference 01: How to get index within for loop?
    # Reference 02: How to pad number with leading zeros?
    for filename in filenames:
        
        # If and only if they are csv files
        if filename.endswith('.csv'):
            number_of_csv_file += 1
            
            # Print to terminal to show user what files is found
            print('\t' + str(number_of_csv_file).zfill(2) + ': ' + filename)
            all_files.append(data_folder + filename)
            
    # all_files = all_files.sort()
            
    # Warning message if you forgot to download all 12 months.
    # For our project we want exactly 12 file
    if number_of_csv_file != 12:
        print('Sorry, I think you are missing some files?')
        
        # Attempt to safely exit the program
        raise SystemExit


# In[5]:


# Creating a list of all CSV file in the path directory
# all_files = glob.glob(data_folder + "/*.csv")

# tempCitibike list that store the file data frame
tempCitibike = []

# Track how many records total
totalRecordCount = 0

# Loop and append the dataframes into tempCitibke List
for filename in sorted(all_files):
    # Filename is the path/file.ext
    print(os.path.basename(filename), end=" ")
    
    # df contains the entire dataset (CSV)
    df = pd.read_csv(filename) 
    
    # How to filter by values
    # https://stackoverflow.com/q/17071871

    # Experiment to see if only using annual membership will show a different
    # df = df.loc[df['usertype'] == 'Subscriber']
    
    # Radnomly select 10% of the data, with random_state = 1
    df = df.sample(frac = 0.1, replace = True, random_state = 1)
    
    # Output the (rowCount, columnCount) to terminal
    print("size: " + str(df.shape))
    
    totalRecordCount += df.shape[0]
    tempCitibike.append(df)

# Combine the tempCitibike dfs together
citibike = pd.concat(tempCitibike)
print("\nThere are " + "{:,}".format(totalRecordCount) + " number of records.")
print("Before dropping NaN: " + "{:,}".format(citibike.shape[0]))

# Drop rows with any (missing) variables
# Improve performance by not using inplace option
# Reference: https://stackoverflow.com/q/22532302
citibike = citibike.dropna()

print("After dropping NaN:  " + "{:,}".format(citibike.shape[0]))


# In[6]:


# Sample my dataset
citibike.head()


# # Question 01: What is monthly total ridership in 2018?

# In[7]:


# What is the type of the starttime?
print('Type: ' + str(type(citibike['starttime'].values[0])))
print(citibike['starttime'].values[0])


# In[8]:


# Need to convert start_time and stop_time to datetime
from datetime import datetime

formatString = '%Y-%m-%d %H:%M:%S.%f'

# Reference 03: Convert datetime column into 
citibike['starttime'] = pd.to_datetime(citibike['starttime'], format=formatString)
citibike['stoptime'] = pd.to_datetime(citibike['stoptime'], format=formatString)


# In[9]:


# Confirm such convertion work, should be datetime64!
print('Type: ' + str(type(citibike['starttime'].values[0])))
print(citibike['starttime'].values[0])


# In[10]:


# Reference 03: Group by month on datetime columns
# Create another pandas dataframe that only has 1 columns
citibike_monthly_df = citibike.filter(['starttime']) # Monthly dataframe
citibike_daily_df = citibike.filter(['starttime'])   # Daily dataframe

# Each of the starttime is now converted into YYYY-MM
citibike_monthly_df.starttime = pd.to_datetime(citibike_monthly_df.starttime).dt.to_period('m')
citibike_daily_df.starttime = pd.to_datetime(citibike_daily_df.starttime).dt.to_period('d')

# Confirm if they are correct
citibike_daily_df.head()
citibike_monthly_df.head()


# In[11]:


# Get frequency count of ridership data by month and day
monthly_answer = citibike_monthly_df.groupby(['starttime']).size()
daily_answer = citibike_daily_df.groupby(['starttime']).size()


# In[12]:


type(monthly_answer)


# In[37]:


ax = monthly_answer.plot(
    figsize = (20,10), 
    kind = 'bar',
    title = '2018 Monthly Citibike Ridership Count',
    fontsize = 12,
    rot=0)

# Finally learned how to do it, you have to assign it to an object first.
# Reference: https://stackoverflow.com/a/21487560

# Then use set_xlable ...
ax.set_xlabel('')

# Very complicated, 
# Reference: https://stackoverflow.com/a/25973637
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y), ',')))


# In[14]:


temperature_dict = {
    'Month':['2018-01', 
             '2018-02',
             '2018-03',
             '2018-04',
             '2018-05',
             '2018-06',
             '2018-07',
             '2018-08',
             '2018-09',
             '2018-10',
             '2018-11',
             '2018-12'],
    'Temperature':[32.5, 35.5, 42.5, 53, 62.5, 71.5, 76.5, 75.5, 68, 57, 48, 37.5]
}

temperature_df = pd.DataFrame.from_dict(temperature_dict)

temperature_df.plot(
    figsize = (20,10), 
    kind = 'line',
    title = '2018 Average Temperature',
    fontsize = 12,
    linewidth = 4.0, 
    markeredgewidth = 10.0,
    linestyle = '--', 
    marker = 'o',
    rot = 0)


# In[15]:


daily_answer.plot(
    figsize = (20,10), 
    kind = 'bar',
    title = '2018 Monthly Citibike Ridership Count',
    fontsize = 12,
    legend=False)


# # Question 02: Pending

# In[16]:


# Enter Python & Pandas Code here :)


# # Question 03: What is the most popular station?

# In[17]:


# Most busy start station
# https://stackoverflow.com/q/29836477
# http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
key = 'start station name'
start_station_df = citibike.groupby(key)[key].count().reset_index(name = 'count')
start_station_df.sort_values(by = 'count', ascending = False, inplace = True)

# The most popular start stations award goes to...
print('The most popular start station: ' + str(start_station_df.iloc[0][0]), end = '')
print(' ({:,} times)'.format(start_station_df.iloc[0][1]))

# Most busy end station
key = 'end station name'
end_station_df = citibike.groupby(key)[key].count().reset_index(name = 'count')
end_station_df.sort_values(by = 'count', ascending = False, inplace = True)

print('The most popular end station:   ' + str(end_station_df.iloc[0][0]), end = '')
print(' ({:,} times)'.format(end_station_df.iloc[0][1]))

# Some information about this station
print('\nFrom Wiki: Pershing Square is a public square in Manhattan, New York City. ')
print('Located where Park Avenue and 42nd Street intersect in front of Grand Central Terminal.')
print('The main roadway of Park Avenue crosses over 42nd Street on the Park Avenue Viaduct. ')
print('It is also known as the Pershing Square Viaduct.')

# Top 10 one way direction...
# Chart Annual rider vs. Weekly rider vs. Single rider

# School station analysis (Where do they go?)
# Male and Female trips
# Age Group

# Longest trip (Distance formula)
# Shortest trip (Distance formula)
# Holiday trip (Federal Holiday)

# Number of conected trip ...
# Start and end station by the same user?

# Analysis on commute habbit
# Year of Birth & Gender (Start time and station)
# Morning starting on the train station and ending on the same station


# In[18]:


# https://stackoverflow.com/q/54009788
# This analysis exclude the trip where they start and end at the same station
# Because it is actually central park, I am guessing for tourist?

# Do not want the trip start and end at the same place
citibike_trip = citibike[citibike['start station name'] != citibike['end station name']]

# Create a subset dataframe with only start & end station name
citibike_trip = citibike_trip.filter(['start station name', 'end station name'])

# Create the trip string
citibike_trip['trip'] = citibike_trip['start station name'] + ' to ' + citibike_trip['end station name']

# Group by trip and add count column
trip_df = citibike_trip.groupby('trip')['trip'].count().reset_index(name = 'count')
trip_df.sort_values(by = 'count', ascending = False, inplace = True)

# Print out the result
print('The most popular ride is : ' + trip_df.iloc[0][0], end = '')
print(' ({} times)'.format(trip_df.iloc[0][1]))
print('The least popular ride is: ' + trip_df.iloc[-1][0], end = '')
print(' ({} times)'.format(trip_df.iloc[-1][1]))


# # Top 10 poupar trips in 2018

# In[19]:


# Take a look at the top 10 popular trips
trip_df.head(10)


# # What are the type of membership (Annual members or paid per ride?)

# In[20]:


membership_df = citibike.groupby('usertype')['usertype'].count()
membership_df = membership_df.reset_index(name = 'count')
membership_df = membership_df.set_index(['usertype'])

# Set the index so it will print properly
membership_df


# In[22]:


# Reference: https://markhneedham.com/blog/2018/09/18/matplotlib-remove-axis-legend/
membership_df.plot(
    title = '2018 Citibike Rider Membership Data', 
    kind = 'bar',
    rot = 0,
    legend = None)


# In[204]:


gender_df = citibike.groupby('gender')['gender'].count()
gender_df = gender_df.reset_index(name = 'count')
gender_df = gender_df.set_index(['gender'])

# Set the index so it will print properly
gender_df


# In[113]:


# Reference: https://markhneedham.com/blog/2018/09/18/matplotlib-remove-axis-legend/
gender_df.plot(
    title = '2018 Citibike Rider Gender', 
    kind = 'bar',
    rot = 0,
    legend = None)


# In[174]:


# Age Group
# https://stackoverflow.com/a/34683105
age_df = citibike[['birth year']].copy()

# Replace the column names
age_df.columns = age_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Failed attempt...
# age_df['age'] = age_df.apply(lambda row: row['birth_year'])

def get_age(year_of_birth):
    result = 2019 - int(year_of_birth)
    if result >= 0 and result <=80 and result != 50:
        return result
    else:
        return -1

# Calculate a new column by using vectorize method
age_df['age'] = np.vectorize(get_age)(age_df['birth_year'])

age_df = age_df[age_df['age'] != -1]

age_df = age_df.groupby('age')['age'].count()
age_df = age_df.reset_index(name = 'count')
age_df = age_df.set_index(['age'])


# In[175]:


age_df.plot(
    figsize = (20,10), 
    title = '2018 Citibike Rider Age', 
    kind = 'bar',
    rot = 0,
    legend = None)


# In[188]:


gender_df = citibike.groupby('gender')['gender'].count()
gender_df = gender_df.reset_index(name = 'count')
gender_df = gender_df.set_index(['gender'])

# Set the index so it will print properly
gender_df


# In[189]:


# https://stackoverflow.com/a/43211266
from math import sin, cos, sqrt, atan2, radians


def get_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in miles 
    R = 3958.8

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def get_duration(datetime1, datetime2):
    # https://stackoverflow.com/a/14920923
    duration = datetime2 - datetime1
    return duration.total_seconds()
    
def get_speed(distance, duration):
    hour = 60.0 * 60.0 * 60.0 / (seconds)
    
    # return distance per hour
    return (distance/hour)

ans = get_distance(40.729538, -73.984267, 40.736197, -74.008592)
ans = get_distance(40.730260, -73.953940, 40.729060, -73.957790)

print("Result:", ans)


# In[199]:


import numpy as np

df = citibike.sample(frac = 0.1, replace = True, random_state = 1)

# Vectorize: https://stackoverflow.com/q/19914937
# start station latitude, start station longitude
df['distance'] = np.vectorize(get_distance)(df['start station latitude'],
                                     df['start station longitude'],
                                     df['end station latitude'],
                                     df['end station longitude'])



# df['duration'] = np.vectorize(get_duration)(df['starttime'], df['stoptime'])
# df['C'] = (df['B'] - df['A'])/ np.timedelta64(1, 's')
df['duration'] = (df['stoptime'] - df['starttime']) / np.timedelta64(1, 's')

df.head()


# In[200]:


df['speed'] = (df['distance'] / ((60.0 * 60.0) / df['duration']))


# In[201]:


# citibike_trip = citibike[citibike['start station name'] != citibike['end station name']]
commute_trip = df[df['start station name'] != df['end station name']]


# In[202]:


commute_trip = commute_trip[(commute_trip['speed'] >= 3.0)
                            & (commute_trip['speed'] <= 25)]

# commute_trip = commute_trip[commute_trip['birth year'] >= 1989]


# In[203]:


commute_trip.head()


# In[ ]:


commute_trip = commute_trip.groupby('speed')['speed'].count()
commute_trip = commute_trip.reset_index(name = 'count')
commute_trip = commute_trip.set_index(['speed'])


# In[ ]:


commute_trip.plot(
    figsize = (20,10), 
    title = '2018 Citibike Commuter Speed', 
    kind = 'line',
    rot = 0,
    legend = None)


# In[ ]:


commute_trip

