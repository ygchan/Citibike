# For starting the chrome application
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# For wait functions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC

import time

# Reference: https://stackoverflow.com/a/49851826
# This is the only way George able to make Chrome() to work on my laptop.
driver = webdriver.Chrome(ChromeDriverManager().install())

citibike_website = 'https://s3.amazonaws.com/tripdata/index.html'

# Open a browser and visit the website
driver.get(citibike_website)
title = driver.title
# wait(driver, 15).until_not(EC.title_is(title))
print(driver.title)

time.sleep(5)

# Reference: https://stackoverflow.com/a/34759880
# This is the way to work around the bs4 limitation to get all links
file_list = driver.find_elements_by_xpath('//a[@href]')

# Iterate through this web object
for file in file_list:
    # This get the https:// link
    link = file.get_attribute("href")
    # Check if this link contains a file
    if (link.endswith('zip')):
    	# Getting the filename part only
    	# 201906-citibike-tripdata.csv.zip
		# 201907-citibike-tripdata.csv.zip
		# 201908-citibike-tripdata.csv.zip
		# ... etc
        print(link.split('/')[-1])