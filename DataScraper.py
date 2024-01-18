from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import csv

url = "https://www.energiemarktinformatie.nl/beurzen/elektra/"

# driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
driver = webdriver.Firefox()

driver.get(url)

wait = WebDriverWait(driver, 4)

# Wait for the library to load (adjust the timeout as needed)
wait.until(
    lambda driver: driver.execute_script("return typeof Highcharts !== 'undefined';")
)

element_to_drag = driver.find_element(By.CLASS_NAME, "highcharts-navigator")

driver.execute_script("arguments[0].scrollIntoView();", element_to_drag)

actions = ActionChains(driver)

# Define the number of steps and the distance to move in each step
num_steps = 13  # Adjust the number of steps for desired slowness
# Adjust the horizontal distance for each step (negative value to move left)
step_x = 100
step_y = 0  # No vertical movement

for _ in range(num_steps):
    actions.click_and_hold(element_to_drag)
    actions.move_by_offset(step_x, step_y).perform()
    actions.release().perform()
    actions.click_and_hold(element_to_drag)
    actions.move_by_offset(step_x, step_y).perform()
    actions.release().perform()
    # Add a small delay between steps (adjust as needed)
    driver.implicitly_wait(4)
    step_x -= 50

WebDriverWait(driver, 4)

for _ in range(num_steps):
    actions.click_and_hold(element_to_drag)
    actions.move_by_offset(step_x, step_y).perform()
    actions.release().perform()
    actions.click_and_hold(element_to_drag)
    actions.move_by_offset(step_x, step_y).perform()
    actions.release().perform()
    # Add a small delay between steps (adjust as needed)
    driver.implicitly_wait(4)
    step_x += 100

# Release the mouse button to end the drag action
actions.release().perform()

data = driver.execute_script(
    """
chartData = [];
Highcharts.charts[0].series[0].data.forEach(function(point){chartData.push([point.x, point.y])});
return chartData;
"""
)

csv_file_path = "data.csv"

with open(csv_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for row in data:
        writer.writerow(row)


print(len(data), data)
print(len(data))
# Close the WebDriver
driver.quit()
