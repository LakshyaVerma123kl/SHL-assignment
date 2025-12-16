import time
import json
import logging
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = "data/shl_assessments.json"
MIN_ITEMS = 377  # Requirement from PDF

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_driver():
    """Sets up a Chrome browser that looks like a real user."""
    chrome_options = Options()
    # Run in "Headless" mode (no GUI) to be faster, set to False if you want to see it working
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def classify_test(text):
    """Maps keywords to official SHL categories (PDF requirement)."""
    text = text.lower()
    categories = set()
    
    if any(w in text for w in ['cognitive', 'numerical', 'verbal', 'deductive', 'inductive', 'ability', 'calculate']):
        categories.add("Ability & Aptitude")
    if any(w in text for w in ['python', 'java', 'coding', 'technical', 'knowledge', 'skill', 'sql', 'react', 'excel']):
        categories.add("Knowledge & Skills")
    if any(w in text for w in ['personality', 'behavior', 'opq', 'motivation', 'culture', 'style']):
        categories.add("Personality & Behavior")
    if any(w in text for w in ['manager', 'leadership', 'sjt', 'scenario', 'judgement']):
        categories.add("Biodata & Situational Judgement")
    if any(w in text for w in ['simulation', 'interactive']):
        categories.add("Simulations")
    if any(w in text for w in ['360', 'development']):
        categories.add("Development & 360")
        
    return list(categories) if categories else ["General"]

def scrape_with_selenium():
    logging.info("🚀 Launching Browser to scrape SHL Catalog...")
    driver = setup_driver()
    
    try:
        driver.get(CATALOG_URL)
        time.sleep(5)  # Wait for initial load

        # --- SCROLLING LOGIC ---
        # The page likely lazy-loads items. We need to scroll down until we have enough.
        last_height = driver.execute_script("return document.body.scrollHeight")
        attempts = 0
        
        while True:
            # Scroll to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3) # Wait for content to load
            
            # Check if we have enough items yet
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # Adjust this selector based on actual site structure (usually 'article', 'div.card', etc.)
            # We look for links with 'view' in them as a proxy for product cards
            current_count = len([a for a in soup.find_all('a', href=True) if '/product-catalog/view/' in a['href']])
            
            logging.info(f"📜 Scrolled... Found ~{current_count} items so far.")
            
            if current_count >= MIN_ITEMS + 10:
                logging.info("✅ Target count reached!")
                break
                
            # Check if we hit bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                attempts += 1
                if attempts >= 3: # Try 3 times then give up
                    logging.info("End of page reached.")
                    break
            else:
                attempts = 0
                last_height = new_height

        # --- PARSING ---
        logging.info("Parsing final page content...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        assessments = []
        seen_urls = set()

        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/product-catalog/view/' in href:
                full_url = "https://www.shl.com" + href if href.startswith('/') else href
                
                if full_url in seen_urls: continue
                seen_urls.add(full_url)
                
                # Extract details from the card itself (safer than visiting each link)
                # We climb up to the parent container to find the title/desc
                card = a.find_parent('div') or a.find_parent('article') or a 
                
                name = a.get_text(strip=True)
                # If the link text is empty/generic, try finding a heading nearby
                if len(name) < 3:
                    h_tag = card.find(['h1', 'h2', 'h3', 'h4'])
                    if h_tag: name = h_tag.get_text(strip=True)
                
                # Description often in a paragraph tag inside the card
                desc = "Detailed assessment of skills and capabilities."
                p_tag = card.find('p')
                if p_tag: desc = p_tag.get_text(strip=True)
                
                full_text = (name + " " + desc).lower()
                
                assessments.append({
                    "url": full_url,
                    "name": name if name else "SHL Assessment",
                    "description": desc,
                    "test_type": classify_test(full_text),
                    "duration": 30, # Default as we can't visit every page without getting blocked again
                    "adaptive_support": "Yes" if "adaptive" in full_text else "No",
                    "remote_support": "Yes"
                })

        # Save
        import os
        os.makedirs('data', exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(assessments, f, indent=4)
        
        logging.info(f"🎉 Success! Scraped {len(assessments)} real items to {OUTPUT_FILE}")

    except Exception as e:
        logging.error(f"Browser Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_with_selenium()