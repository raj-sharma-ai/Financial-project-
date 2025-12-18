
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import csv
from datetime import datetime
import time


def scrape_fund_details(fund_url: str) -> Dict[str, str]:
    """
    Scrapes detailed metrics from individual fund page using exact class structure.
    
    Looks for:
    - div class="Overview_web_tabContent__tgoQU"
    - ul class="OverviewContent_web_categoryWrapper__lQnP9"
    - li items with span class="OverviewContent_web_name__SnE9y" (label)
    - and span class="OverviewContent_web_value__rmJI_" (value)
    
    Args:
        fund_url: URL of the individual fund page
        
    Returns:
        Dictionary with expense ratio, sharpe ratio, std dev, beta with category averages
    """
    details = {
        'expense_ratio': '',
        'sharpe_ratio': '',
        'sharpe_category_avg': '',
        'standard_deviation': '',
        'std_dev_category_avg': '',
        'beta': '',
        'beta_category_avg': ''
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Add small delay to avoid overwhelming the server
        time.sleep(0.5)
        
        response = requests.get(fund_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the Overview tab content div
        overview_div = soup.find('div', class_='Overview_web_tabContent__tgoQU')
        
        if not overview_div:
            print(f"  ⚠ Overview div not found")
            return details
        
        # Find the category wrapper ul
        category_ul = overview_div.find('ul', class_='OverviewContent_web_categoryWrapper__lQnP9')
        
        if not category_ul:
            print(f"  ⚠ Category wrapper ul not found")
            return details
        
        # Find all li items
        li_items = category_ul.find_all('li')
        
        print(f"  📊 Found {len(li_items)} metric items")
        
        for li in li_items:
            # Find name span (label)
            name_span = li.find('span', class_='OverviewContent_web_name__SnE9y')
            # Find value span
            value_span = li.find('span', class_='OverviewContent_web_value__rmJI_')
            
            if name_span and value_span:
                label = name_span.get_text(strip=True)
                value = value_span.get_text(strip=True)
                
                label_lower = label.lower()
                
                # Extract Expense Ratio
                if 'expense ratio' in label_lower:
                    details['expense_ratio'] = value
                    print(f"  ✓ Expense Ratio: {value}")
                
                # Extract Sharpe Ratio / Category Average
                elif 'sharpe ratio' in label_lower:
                    # Value format: "0.61  / 0.75"
                    if '/' in value:
                        parts = value.split('/')
                        details['sharpe_ratio'] = parts[0].strip()
                        details['sharpe_category_avg'] = parts[1].strip() if len(parts) > 1 else ''
                        print(f"  ✓ Sharpe Ratio: {details['sharpe_ratio']} / Category Avg: {details['sharpe_category_avg']}")
                    else:
                        details['sharpe_ratio'] = value
                        print(f"  ✓ Sharpe Ratio: {value}")
                
                # Extract Standard Deviation / Category Average
                elif 'standard deviation' in label_lower:
                    # Value format: "12.5  / 11.8"
                    if '/' in value:
                        parts = value.split('/')
                        details['standard_deviation'] = parts[0].strip()
                        details['std_dev_category_avg'] = parts[1].strip() if len(parts) > 1 else ''
                        print(f"  ✓ Std Dev: {details['standard_deviation']} / Category Avg: {details['std_dev_category_avg']}")
                    else:
                        details['standard_deviation'] = value
                        print(f"  ✓ Standard Deviation: {value}")
                
                # Extract Beta / Category Average
                elif 'beta' in label_lower:
                    # Value format: "0.95  / 1.00"
                    if '/' in value:
                        parts = value.split('/')
                        details['beta'] = parts[0].strip()
                        details['beta_category_avg'] = parts[1].strip() if len(parts) > 1 else ''
                        print(f"  ✓ Beta: {details['beta']} / Category Avg: {details['beta_category_avg']}")
                    else:
                        details['beta'] = value
                        print(f"  ✓ Beta: {value}")
        
    except Exception as e:
        print(f"  ⚠ Error scraping fund details from {fund_url}: {e}")
    
    return details


def scrape_mutual_fund_table(url: str, debug: bool = False, scrape_details: bool = True, max_tables: int = 23) -> List[Dict[str, str]]:
    """
    Scrapes mutual fund data from an HTML table.
    Stops at table 24 (index 23) and automatically saves data.
    
    Args:
        url: The URL of the page containing the mutual fund table
        debug: If True, prints HTML structure for debugging
        scrape_details: If True, scrapes individual fund pages for detailed metrics
        max_tables: Maximum number of tables to process (default 23, stops before table 24)
        
    Returns:
        List of dictionaries containing fund data
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find ALL table bodies (multiple tables on page)
        tbodies = soup.find_all('tbody')
        if not tbodies:
            print("Warning: No <tbody> found in the HTML")
            return []
        
        print(f"Found {len(tbodies)} tables on the page")
        print(f"⚠️  Will stop at table {max_tables + 1} (processing tables 1-{max_tables})")
        
        funds_data = []
        
        # Loop through each table - STOP at table 24
        for table_idx, tbody in enumerate(tbodies, 1):
            # Check if we've reached table 24
            if table_idx > max_tables:
                print(f"\n🛑 STOPPING: Reached table {table_idx}")
                print(f"✓ Processed {table_idx - 1} tables successfully")
                break
            
            print(f"\nProcessing table {table_idx}...")
            
            # Loop through each row in this tbody
            for idx, row in enumerate(tbody.find_all('tr'), 1):
                if debug and table_idx == 1 and idx == 1:
                    print("\n=== DEBUGGING FIRST ROW ===")
                    print(row.prettify())
                    print("\n=== ALL TD ELEMENTS ===")
                    for td_idx, td in enumerate(row.find_all('td')):
                        print(f"TD {td_idx}: class={td.get('class')}, text='{td.get_text(strip=True)[:50]}'")
                        print("=" * 80 + "\n")
                
                try:
                    fund_dict = extract_row_data(row)
                    
                    # Add the fund data regardless of name or null values
                    if fund_dict and fund_dict.get('fund_name'):
                        # If scrape_details is True, fetch detailed metrics
                        if scrape_details and fund_dict.get('fund_link'):
                            full_url = fund_dict['fund_link']
                            
                            # Handle relative URLs
                            if full_url and not full_url.startswith('http'):
                                base_url = 'https://www.moneycontrol.com'
                                full_url = base_url + full_url
                            
                            print(f"  [{idx}] Scraping: {fund_dict['fund_name']}")
                            
                            # Get detailed metrics
                            details = scrape_fund_details(full_url)
                            
                            # Merge details into fund_dict
                            fund_dict.update(details)
                        
                        funds_data.append(fund_dict)
                        
                        # Just log if we find 'Others' but DON'T STOP
                        if 'others' in fund_dict['fund_name'].lower():
                            print(f"  Found 'Others' fund: {fund_dict['fund_name']} - continuing...")
                        
                except Exception as e:
                    print(f"Error processing row {idx} in table {table_idx}: {e}")
                    # Continue to next row even on error
                    continue
        
        print(f"\n✓ Total funds scraped: {len(funds_data)}")
        return funds_data
        
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def extract_row_data(row) -> Dict[str, str]:
    """
    Extracts data from a single table row using TD position.
    Handles null/empty values gracefully.
    """
    tds = row.find_all('td')
    
    if len(tds) == 0:
        return {}
    
    # Try to extract fund name from first TD with an <a> tag
    fund_name = ''
    fund_link = ''
    
    # Search for the first <a> tag in the row (usually in first TD)
    link_tag = row.find('a')
    if link_tag:
        fund_name = link_tag.get_text(strip=True)
        fund_link = link_tag.get('href', '')
    
    # Extract by position - use empty string for missing values
    fund_dict = {
        'fund_name': fund_name,
        'fund_link': fund_link,
        'crisil_rank': tds[1].get_text(strip=True) if len(tds) > 1 else '',
        'aum': tds[2].get_text(strip=True) if len(tds) > 2 else '',
        'return_1m': tds[3].get_text(strip=True) if len(tds) > 3 else '',
        'return_6m': tds[4].get_text(strip=True) if len(tds) > 4 else '',
        'return_1y': tds[5].get_text(strip=True) if len(tds) > 5 else '',
        'return_3y': tds[6].get_text(strip=True) if len(tds) > 6 else '',
        'return_5y': tds[7].get_text(strip=True) if len(tds) > 7 else ''
    }
    
    return fund_dict


def save_to_json(data: List[Dict], filename: str = None):
    """Save scraped data to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mutual_funds_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(data)} funds to {filename}")
    return filename


def save_to_csv(data: List[Dict], filename: str = None):
    """Save scraped data to CSV file."""
    if not data:
        print("No data to save to CSV")
        return None
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mutual_funds_{timestamp}.csv"
    
    # Get all possible keys from all dictionaries
    fieldnames = ['fund_name', 'fund_link', 'crisil_rank', 'aum', 
                  'return_1m', 'return_6m', 'return_1y', 'return_3y', 'return_5y',
                  'expense_ratio', 'sharpe_ratio', 'sharpe_category_avg',
                  'standard_deviation', 'std_dev_category_avg', 
                  'beta', 'beta_category_avg']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Saved {len(data)} funds to {filename}")
    return filename


def main():
    """Main execution function."""
    url = 'https://www.moneycontrol.com/mutualfundindia/'
    
    print(f"Scraping data from: {url}")
    print("-" * 80)
    
    # Run scraper - will automatically stop at table 24
    print("STARTING SCRAPER...")
    print("⚠ This will take longer as we're scraping individual fund pages...")
    print("🛑 Will automatically stop at table 24 and save data")
    print("-" * 80)
    
    # Set max_tables=23 to stop before table 24
    # Set scrape_details=True to scrape individual pages
    # Set scrape_details=False to only scrape the main table (faster)
    funds_data = scrape_mutual_fund_table(url, debug=False, scrape_details=True, max_tables=23)
    
    # ALWAYS save data before any processing or stopping
    if funds_data:
        print("\n" + "=" * 80)
        print("AUTOMATICALLY SAVING DATA...")
        print("=" * 80)
        
        # Save to both JSON and CSV
        json_file = save_to_json(funds_data)
        csv_file = save_to_csv(funds_data)
        
        print(f"\n✓ Successfully scraped and saved {len(funds_data)} mutual funds")
        print(f"✓ JSON file: {json_file}")
        print(f"✓ CSV file: {csv_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Count funds with 'Others' in name
        others_count = sum(1 for f in funds_data if 'others' in f['fund_name'].lower())
        print(f"Total funds: {len(funds_data)}")
        print(f"Funds with 'Others' in name: {others_count}")
        
        # Count funds with detailed metrics
        with_expense_ratio = sum(1 for f in funds_data if f.get('expense_ratio'))
        with_sharpe_ratio = sum(1 for f in funds_data if f.get('sharpe_ratio'))
        with_std_dev = sum(1 for f in funds_data if f.get('standard_deviation'))
        with_beta = sum(1 for f in funds_data if f.get('beta'))
        
        print(f"Funds with Expense Ratio: {with_expense_ratio}")
        print(f"Funds with Sharpe Ratio: {with_sharpe_ratio}")
        print(f"Funds with Standard Deviation: {with_std_dev}")
        print(f"Funds with Beta: {with_beta}")
        
        # Print first 3 funds with details
        print("\nFirst 3 funds with detailed metrics:")
        for idx, fund in enumerate(funds_data[:3], 1):
            print(f"\n{idx}. {fund['fund_name']}")
            print(f"   Returns: 1M={fund['return_1m']}, 6M={fund['return_6m']}, 1Y={fund['return_1y']}")
            print(f"   Expense Ratio: {fund.get('expense_ratio', 'N/A')}")
            print(f"   Sharpe Ratio: {fund.get('sharpe_ratio', 'N/A')} (Category Avg: {fund.get('sharpe_category_avg', 'N/A')})")
            print(f"   Std Deviation: {fund.get('standard_deviation', 'N/A')} (Category Avg: {fund.get('std_dev_category_avg', 'N/A')})")
            print(f"   Beta: {fund.get('beta', 'N/A')} (Category Avg: {fund.get('beta_category_avg', 'N/A')})")
        
        print(f"\n... and {len(funds_data) - 3} more funds" if len(funds_data) > 3 else "")
        
    else:
        print("❌ No data scraped. Check the URL and website structure.")
    
    return funds_data


if __name__ == "__main__":
    funds_list = main()
    
    # Print confirmation
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE!")
    print("=" * 80)
    print(f"Total records collected: {len(funds_list)}")
    print("✓ Data has been automatically saved to JSON and CSV files!")
    print("🛑 Stopped at table 24 as requested")
    print("\nNote: Some funds may not have all metrics available on their pages.")