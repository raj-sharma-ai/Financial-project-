import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import time
import csv

class HealthInsuranceScraper:
    def __init__(self):
        # List of valid policy URLs only
        self.policies = [
            {"insurer": "Aditya Birla Capital", "policy_name": "Activ One", "url": "https://www.adityabirlacapital.com/healthinsurance/activ-one"},
            {"insurer": "HDFC ERGO", "policy_name": "Optima Secure", "url": "https://www.hdfcergo.com/health-insurance/optima-secure"},
            {"insurer": "ManipalCigna", "policy_name": "Sarvah", "url": "https://www.manipalcigna.com/hospitalization-cover/manipalcigna-sarvah"},
            {"insurer": "Tata AIG", "policy_name": "MediCare Plus", "url": "https://www.tataaig.com/health-insurance/medicare-plus"},
            {"insurer": "Acko", "policy_name": "Acko Health", "url": "https://www.acko.com/health-insurance/health-insurance-plans-for-family/"},
            {"insurer": "Zurich Kotak", "policy_name": "Health Premier Plan", "url": "https://www.zurichkotak.com/services/health-insurance/health-super-top-up"},
            {"insurer": "Niva Bupa", "policy_name": "Health Companion", "url": "https://www.nivabupa.com/family-health-insurance-plans/health-companion-family-floater.html"}
        ]
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    def fetch_page(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ").lower()
            return soup, text
        except Exception as e:
            print(f":x: Failed to fetch {url}: {e}")
            return None, None

    def extract_sum_insured(self, text):
        match = re.search(r'₹\s*(\d+(?:,\d+)*)\s*[lL]akh\s*(?:to|-)?\s*₹?\s*(\d+(?:,\d+)*)\s*([lL]akh|[cC]rore)', text)
        if match:
            return f"₹{match.group(1)} Lakh–₹{match.group(2)} {match.group(3).capitalize()}"
        return "Check policy"

    def extract_premium_range(self, text):
        match = re.search(r'₹\s*(\d+(?:,\d+)*)\s*(?:per|/)\s*(?:year|annum)', text)
        if match:
            return f"Starts from ₹{match.group(1)}"
        return "Varies by age & sum insured"

    def detect_boolean(self, text, terms):
        return any(term in text for term in terms)

    def extract_unique_features(self, soup):
        features = []
        for li in soup.select("ul li"):
            content = li.get_text(strip=True).lower()
            if any(keyword in content for keyword in ["worldwide", "restoration", "modern", "mental", "wellness"]):
                features.append(li.get_text(strip=True))
        return features[:10]

    def parse_policy(self, policy):
        print(f":arrow_forward: Scraping {policy['policy_name']} …")
        soup, text = self.fetch_page(policy["url"])
        if not soup or not text:
            print(f":x: Skipped {policy['policy_name']} due to fetch error")
            return None
        data = {
            "policy_name": policy["policy_name"],
            "insurer": policy["insurer"],
            "url": policy["url"],
            "premium_amount_range": self.extract_premium_range(text),
            "sum_insured_range": self.extract_sum_insured(text),
            "waiting_period_general": "30 days" if "30 days" in text else "Check policy",
            "waiting_period_preexisting": "24–48 months" if any(x in text for x in ["24 months", "36 months", "48 months"]) else "Check policy",
            "covers_preexisting": self.detect_boolean(text, ["pre-existing", "ped"]),
            "maternity_cover": self.detect_boolean(text, ["maternity", "pregnancy"]),
            "critical_illness_cover": self.detect_boolean(text, ["critical illness"]),
            "hospital_cash_benefit": self.detect_boolean(text, ["hospital cash"]),
            "daycare_procedures_covered": "Yes" if "day care" in text else "No",
            "opd_cover": self.detect_boolean(text, ["opd"]),
            "ambulance_cover": "Yes" if "ambulance" in text else "No",
            "annual_health_checkup": "Yes" if "health check" in text else "No",
            "network_hospitals": "Wide network",
            "no_claim_bonus": "Yes" if "no claim bonus" in text else "No",
            "automatic_restore": "Yes" if "restore" in text else "No",
            "unique_features": self.extract_unique_features(soup),
            "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return data

    def run(self):
        all_data = []
        for policy in self.policies:
            data = self.parse_policy(policy)
            if data:
                all_data.append(data)
            time.sleep(2)  # polite delay

        # Save JSON
        with open("health_policies.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(":heavy_check_mark: JSON saved: health_policies.json")

        # Save CSV
        with open("health_policies_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Policy Name","Insurer","URL","Premium Range","Sum Insured","Waiting Period General",
                             "Waiting Period Preexisting","Covers Preexisting","Maternity Cover","Critical Illness Cover",
                             "Hospital Cash","Daycare Covered","OPD Cover","Ambulance Cover","Annual Health Checkup",
                             "Network Hospitals","No Claim Bonus","Automatic Restore","Unique Features","Scraped At"])
            for p in all_data:
                writer.writerow([
                    p["policy_name"], p["insurer"], p["url"], p["premium_amount_range"], p["sum_insured_range"],
                    p["waiting_period_general"], p["waiting_period_preexisting"], p["covers_preexisting"],
                    p["maternity_cover"], p["critical_illness_cover"], p["hospital_cash_benefit"],
                    p["daycare_procedures_covered"], p["opd_cover"], p["ambulance_cover"], p["annual_health_checkup"],
                    p["network_hospitals"], p["no_claim_bonus"], p["automatic_restore"], " | ".join(p["unique_features"]),
                    p["scraped_at"]
                ])
        print(":heavy_check_mark: CSV saved: health_policies_summary.csv")


if __name__ == "__main__":
    scraper = HealthInsuranceScraper()
    scraper.run()