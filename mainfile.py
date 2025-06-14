import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from googlesearch import search  # pip install googlesearch-python

# NLI model for claim extraction and analysis
nli = pipeline("text-classification", model="roberta-large-mnli")

# Factual Reporting score from mbfc website
def fetch_mbfc_factual_score(domain):
    search_url = f"https://mediabiasfactcheck.com/?s={domain}"
    try:
        resp = requests.get(search_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        link = soup.find("a", href=True, text=lambda t: t and domain in t.lower())
        if not link:
            return "Unknown"

        page = requests.get(link["href"], timeout=10).text
        psoup = BeautifulSoup(page, "html.parser")
        text = psoup.get_text()

        label_score = {
            "Very High": 10,
            "High": 8,
            "Mostly Factual": 6,
            "Mixed": 4,
            "Low": 2,
            "Very Low": 1
        }

        for label, score in label_score.items():
            if f"Factual Reporting: {label}" in text:
                return score
        return "Unknown"
    except Exception as e:
        print(f"Error fetching MBFC score for {domain}: {e}")
        return "Unknown"

# Web scraping
def scrape_website(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service("./chromedriver.exe"), options=options)
    try:
        driver.get(url)
        time.sleep(7)
        return driver.page_source
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""
    finally:
        driver.quit()

def clean_body_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip())

# NLI stance detection
def get_nli_stance(claim, article_text):
    try:
        result = nli({
            "premise": article_text[:1000],  # truncating to keep within token limit
            "hypothesis": claim
        })[0]
        label = result["label"]
        return {
            "ENTAILMENT": "Agree",
            "CONTRADICTION": "Disagree",
            "NEUTRAL": "Neutral"
        }.get(label, "Neutral")
    except Exception as e:
        print(f"NLI error: {e}")
        return "Neutral"

# Claim processing
def process_claim(claim, num_sites=5):
    results = []
    print(f"\n🔎 Searching for claim: {claim}")
    query = f"{claim} site:news"
    urls = list(search(query, num_results=num_sites))

    for url in urls:
        print(f"🌐 Analyzing: {url}")
        html = scrape_website(url)
        if not html:
            continue
        dom_content = clean_body_content(html)
        stance = get_nli_stance(claim, dom_content)
        domain = urlparse(url).netloc
        credibility_score = fetch_mbfc_factual_score(domain)
        results.append({
            "url": url,
            "domain": domain,
            "stance": stance,
            "credibility_score": credibility_score
        })

    # Save to CSV
    with open("claim_analysis.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "domain", "stance", "credibility_score"])
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Results saved to claim_analysis.csv")
    return results

# Run
if __name__ == "__main__":
    input_claim = input("Enter the claim to verify: ")
    process_claim(input_claim)
