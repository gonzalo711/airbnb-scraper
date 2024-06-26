#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import pandas as pd
import calendar
from urllib.parse import urlparse, parse_qs
from google.cloud import storage




def get_interval_dates(year, month):
    intervals = []
    _, days_in_month = calendar.monthrange(year, month)

    # Utility to adjust check-out date, ensuring it doesn't exceed the month's last day
    def adjust_check_out(check_in, added_days):
        check_out = check_in + timedelta(days=added_days)
        if check_out.month != check_in.month:
            return datetime(check_in.year, check_in.month, days_in_month)
        return check_out

    # Find the first Monday and first Friday of the month
    date = datetime(year, month, 1)
    while date.weekday() != 0:  # Loop until Monday is found
        date += timedelta(days=1)
    first_monday = date

    date = datetime(year, month, 1)
    while date.weekday() != 4:  # Loop until Friday is found
        date += timedelta(days=1)
    first_friday = date

    # Generate weekday stays (Monday to Friday)
    current_date = first_monday
    while current_date.month == month:
        check_in = current_date
        check_out = adjust_check_out(check_in, 4)  # Weekdays: Add 4 days
        intervals.append((check_in, check_out, 'weekday'))
        current_date += timedelta(weeks=1)

    # Generate weekend stays (Friday to Sunday)
    current_date = first_friday
    while current_date.month == month:
        check_in = current_date
        check_out = adjust_check_out(check_in, 2)  # Weekends: Add 2 days
        intervals.append((check_in, check_out, 'weekend'))
        current_date += timedelta(weeks=1)

    # Sort intervals by check-in date to maintain chronological order
    intervals.sort(key=lambda x: x[0])

    return intervals

def construct_urls(dates, bedrooms_guests_mapping):
    base_url = 'https://www.airbnb.com/s/2nd-arrondissement--Paris--France/homes'
    urls = []
    for bedrooms, guests in bedrooms_guests_mapping.items():
        for start_date, end_date, interval_type in dates:  # Adjusted to unpack three items
            url = f"{base_url}?adults={guests}&min_bedrooms={bedrooms}&checkin={start_date.strftime('%Y-%m-%d')}&checkout={end_date.strftime('%Y-%m-%d')}&room_types%5B%5D=Entire%20home%2Fapt"
            urls.append(url)
    return urls


def scrape_data(driver, url, competitors_ids,livinparis_ids, max_pages=5):
    
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    listings = []
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    check_in_date = query_params.get('checkin', [''])[0]
    check_out_date = query_params.get('checkout', [''])[0]
    bedrooms = query_params.get('min_bedrooms', [''])[0]  
    

    current_page = 1
    while current_page <= max_pages:
                
        print(f"Scraping page {current_page}")
        time.sleep(2)
            
        # Scroll down to the bottom of the page to trigger dynamic content loading
        body = driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.END)
            
        # Wait a bit for any dynamically loaded content to appear
        time.sleep(3)
        
        try:
            
                       
            # Wait for listings to be visible
            wait.until(EC.visibility_of_all_elements_located((By.XPATH, '//div[@data-testid="listing-card-title"]')))

            # Scraping logic here (adapted to your current scraping needs)
            # Example:
            titles = driver.find_elements(By.XPATH, '//div[@data-testid="listing-card-title"]')
            subtitles = driver.find_elements(By.XPATH, '//div[@data-testid="listing-card-name"]')
            details = driver.find_elements(By.XPATH, '//div[@data-testid="listing-card-subtitle"]')
            prices = driver.find_elements(By.XPATH, '//div[@class="_tt122m"]')
            urls = driver.find_elements(By.XPATH, '//div[@class="cy5jw6o atm_5j_8todto atm_70_87waog atm_j3_1u6x1zy atm_jb_4shrsx atm_mk_h2mmj6 atm_vy_7abht0  dir dir-ltr"]/a')
            ratings_and_reviews = driver.find_elements(By.XPATH, '//span[@class="ru0q88m atm_cp_1ts48j8 dir dir-ltr"]')
            superhosts = driver.find_elements(By.XPATH, '//div[@class="t1qa5xaj dir dir-ltr"]')
            
            for i, title in enumerate(titles):
                listing_url = urls[i].get_attribute('href')
                listing_id = listing_url.split("/")[-1].split("?")[0]
                competitor = "Yes" if listing_id in competitors_ids else "No"
                livinparis = "Yes" if listing_id in livinparis_ids else "No"
                    
                listings.append({
                    'Title': title.text,
                    'Subtitle': subtitles[i].text if i < len(subtitles) else 'N/A',
                    'Detail': details[i].text if i < len(details) else 'N/A',
                    'Price': prices[i].text if i < len(prices) else 'N/A',
                    'URL': listing_url,
                    'Bedrooms': bedrooms,
                    'Review_rating': ratings_and_reviews[i].text if i < len(ratings_and_reviews) else 'N/A',
                    'Superhost': superhosts[i].text if i < len(superhosts) else 'N/A',
                    'Check_in': check_in_date,
                    'Check_out': check_out_date,
                    'Competitor': competitor,
                    'Livinparis':livinparis,
                    'Listing_id': listing_id
                    })

            # Attempt to navigate to the next page
            next_page_buttons = driver.find_elements(By.CSS_SELECTOR, 'a[aria-label="Next"]')
            if next_page_buttons:
                try:
                    wait.until(EC.element_to_be_clickable(next_page_buttons[0]))
                    driver.execute_script("arguments[0].click();", next_page_buttons[0])
                    time.sleep(3)  # Give time for the page to load
                    current_page += 1
                except Exception as e:
                    print('Failed to navigate to the next page:', e)
                    break
            else:
                print("No more pages to scrape.")
                break

        except TimeoutException:
            print(f"Timed out waiting for page {current_page} to load.")
            break
        except NoSuchElementException:
            print(f"Missing element on page {current_page}.")
            break
        except Exception as e:
            print(f"An unexpected error occurred on page {current_page}: {e}")
            break

    return listings    



def upload_to_gcs(bucket_name, source_file_names):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for source_file_name in source_file_names:
        destination_blob_name = os.path.basename(source_file_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def main():
    
    selenium_grid_url = os.environ.get('SELENIUM_GRID_URL', 'http://selenium-grid:4444/wd/hub')
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Remote(
        command_executor=selenium_grid_url,
        desired_capabilities=DesiredCapabilities.CHROME,
        options=options
    )

    # Calculate the next three months
    today = datetime.today()
    months_to_scrape = [(today.year, today.month + i) for i in range(1, 4)]
    months_to_scrape = [(year + month // 12, month % 12 if month % 12 != 0 else 12) for year, month in months_to_scrape]

    bedrooms_guests_mapping = {2: 6, 3: 8, 4: 10}
    csv_filenames = []
    
    
    competitors_ids = ["1067185846882110336", "1056273391664839516", "2535408", "923083442903877154", "23708945", 
    "28972065", "49644497", "33215018", "888262574509626431", "541774648085970947", "36069798", 
    "943300723336375806", "948470727933914785", "968528007655958469", "954104853456855150", 
    "989588663736140176", "958720565448264526", "833963940349663276", "21468838", "843192767245443038", 
    "13330999", "36158191", "644660014494289160", "52801631", "34380589", "35186889", "33473475", 
    "54354250", "41757253", "36678564", "764178593487575716", "49184174", "53503892", "45859028", 
    "5817926", "827417087678773788", "5036597", "8136227", "6407918", "24485025", "49186538", 
    "1039081914404298900", "785284068115460701", "696859416537324166", "864923431673683009", "46308484", 
    "40504697", "44975345", "26486469", "862281812777253778", "16751892", "45654615", 
    "659080349297484746", "12133376", "986757474785369429", "986094158438117652", "40772860", 
    "708541033902709139", "36537429", "17432320", "1020195654453354901", "1031804837352534349", 
    "891254912351960811", "15079414", "815930572303320022", "827328195660007964", "51793427", 
    "43541768", "40170305", "41738151", "996067699380811503", "986002929373862221", 
    "909197222612433186", "978659831930304606", "50153725", "933763378760022624", "877513104567993460", 
    "904317794167423250", "975259241354860844", "957774221191844095", "873017003912847803", 
    "706866991374692482", "18818753", "38798785", "845446084482820460", "623704090072982924", 
    "868812389175200997", "897816712189862960", "50261654", "944150258321530571", "708629734607400719", 
    "574364646781540870", "52151320", "7041794", "594675982427827274", "944085541406962003", 
    "730809825518057035", "52889380", "20755411", "999069406606750061", "39483620", "45886519", 
    "21011314", "43937082", "36307456", "562908323845309613", "41373139", "1014544768450877013", 
    "943271155800619740", "1003882289027034269", "1020128430220989045", "42185300", 
    "952953840207260555", "573242429250599167", "948312536432087948", "948366734080037976", 
    "46001713", "19875609", "11196371", "928028073319521423", "964313120505248350", 
    "1038550168200142322", "998953625406772138", "893998484038590240", "1052202590499595067", 
    "19280232", "978866370139112379", "47307307", "49186423", "721395224888573438", 
    "612080162106232567", "844041197461703554", "932607956634548337", "43635890", "16852946", 
    "976646637355580536", "955568495054255979", "53932916", "884024533730371874", "30858454", 
    "918033845689093380", "983821395195091924", "650474249333817505", "974549133785183538", 
    "667252505253500559", "867209627146280811", "724515188109588042", "793283626318010574", 
    "50718454", "872140231900255014", "37698396", "719994888358485259", "792607630544924055", 
    "44441869", "900860219983207401", "663432873412388667", "983698500409593710", "43502575", 
    "999129868617373318", "37618657", "36450873", "53530344", "1011336899500745915", "39726305", 
    "1007756510384280776", "858013771449803961", "39754983", "950491044044564855", 
    "807573377115971455", "662644689793547326", "609106640789714412", "742587092103722023", 
    "19684395", "53490856", "30494526", "43938941", "902791752502933479", "50891362", 
    "797036901085794734", "4019156", "26073789", "40833993", "663540070001854308", "52890054", 
    "34380017", "33036263", "32605058", "40171302", "39851605", "52317072", "43974611", 
    "1011447704766845501", "724515195294356483", "24662794", "13670403", "970724713956277226", 
    "54035270", "664272277980666480", "19358943", "998362185133160001", "919320653251008138", 
    "930909044410593072", "19518859", "552595002753715860", "50317142", "946269737846168408", 
    "771544511970057205", "51780412", "27080053", "12697519", "773052174164998718", 
    "577939173957247868", "686531826462022081", "43772176", "649811895643"]
    
    livinparis_ids= ["12584340","12584603","41576887","41576801","41576693","41576606","725904212179042294",
    "12584693","12584802","560547206045849269","27747787","13386060","13386198","13386132","27746265",
    "25155031","41576944","41574523","944976077648948227","13385873","23055173","40052769","41574646",
    "13385996","41573906","27747200","1068759626559812433","42870861","41575233","1084698515373405958",
    "19230958","19263737","25155033"]

    for year, month in months_to_scrape:
        print(f"Scraping for {calendar.month_name[month]} {year}")
        interval_dates = get_interval_dates(year, month)
        all_urls = construct_urls(interval_dates, bedrooms_guests_mapping)

        all_listings = []
        for url in all_urls:
            print(f"Scraping URL: {url}")
            listings_data = scrape_data(driver, url, competitors_ids, livinparis_ids, max_pages=5)
            all_listings.extend(listings_data)

        # Save to CSV
        df = pd.DataFrame(all_listings)
        csv_filename = f"airbnb_final_listings_{year}_{month}.csv"
        df.to_csv(csv_filename, index=False)
        csv_filenames.append(csv_filename)  # Add the filename to the list
        print(f"Data saved to {csv_filename}")
    
    # After scraping and saving all CSVs, upload them to GCS
    upload_to_gcs('us-central1-airbnbcomposer-b06b3309-bucket', csv_filenames)

    driver.quit()

if __name__ == "__main__":
    main()  

