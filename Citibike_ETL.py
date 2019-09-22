from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# Reference: https://stackoverflow.com/a/49851826
# This is the only way George able to make Chrome() to work on my laptop.
driver = webdriver.Chrome(ChromeDriverManager().install())

citibike_website = 'https://s3.amazonaws.com/tripdata/index.html'

title = driver.title

# Open a browser and visit the website
driver.get(citibike_website)
wait(driver, 15).until_not(EC.title_is(title))
print('Page Loaded!')

actions = ActionChains(driver)
actions.key_down(Keys.CONTROL)
actions.send_keys('a')
actions.key_up(Keys.CONTROL)
actions.perform()

actions.key_down(Keys.CONTROL)
actions.send_keys('c')
actions.key_up(Keys.CONTROL)
actions.perform()

print('Done!')