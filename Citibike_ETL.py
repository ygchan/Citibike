from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

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

time.sleep(10)

# Click the title once
driver.find_element_by_name('h1-title').click()

# Send the command A
ActionChains(driver).key_down(Keys.COMMAND)
time.sleep(2)
ActionChains(driver).send_keys('a')
time.sleep(2)
ActionChains(driver).key_up(Keys.COMMAND)
time.sleep(2)
ActionChains(driver).perform()

print('Send Command A...')

# Send the command C
time.sleep(2)
ActionChains(driver).key_down(Keys.COMMAND)
time.sleep(2)
ActionChains(driver).send_keys('c')
time.sleep(2)
ActionChains(driver).key_up(Keys.COMMAND)
time.sleep(2)
ActionChains(driver).perform()

print('Send Command C...')
print('Done!')