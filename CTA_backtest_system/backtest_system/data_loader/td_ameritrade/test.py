# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager,ChromeType
# from tda import auth, client
# import json

# driver = webdriver.Chrome(ChromeDriverManager(
#     chrome_type=ChromeType.CHROMIUM).install())

# driver.get("http://www.python.org")



# token_path = '/path/to/token.json'
# api_key = '6BVNKUMULSPGMLZU5OUR5JGJMNNDIKHZ'
# redirect_uri = 'https://your.redirecturi.com'
# try:
#     c = auth.client_from_token_file(token_path, api_key)
# except FileNotFoundError:
#     from selenium import webdriver
#     with webdriver.Chrome() as driver:
#         c = auth.client_from_login_flow(
#             driver, api_key, redirect_uri, token_path)

# r = c.get_price_history('AAPL',
#         period_type=client.Client.PriceHistory.PeriodType.YEAR,
#         period=client.Client.PriceHistory.Period.TWENTY_YEARS,
#         frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
#         frequency=client.Client.PriceHistory.Frequency.DAILY)
# assert r.status_code == 200, r.raise_for_status()
# print(json.dumps(r.json(), indent=4))

	
import time

from selenium import webdriver



driver = webdriver.Chrome('/path/to/chromedriver')  # Optional argument, if not specified will search path.

driver.get('http://www.google.com/');

time.sleep(5) # Let the user actually see something!

search_box = driver.find_element_by_name('q')

search_box.send_keys('ChromeDriver')

search_box.submit()

time.sleep(5) # Let the user actually see something!

driver.quit()