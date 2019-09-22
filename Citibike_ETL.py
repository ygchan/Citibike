from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Reference: https://stackoverflow.com/a/49851826
# This is the only way George able to make Chrome() to work on my laptop.
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('http://google.com')