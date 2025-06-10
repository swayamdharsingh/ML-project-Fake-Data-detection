import csv
import time
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from googlesearch import search  # pip install googlesearch-python


llm = Ollama(model="codellama")

# prompt
template = (
    "You are given a claim: \"{claim}\" and a webpage content: \"{dom_content}\".\n"
    "Based on the content, determine whether the webpage agrees, disagrees, or is neutral about the claim.\n"
    "Respond with only one word: Agree, Disagree, or Neutral."
)
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# scraping
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


def clean_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# web searching for similar sites and comparing
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
        dom_content = clean_body_content(html)[:6000]  # Truncate for LLM input
        result = chain.invoke({"claim": claim, "dom_content": dom_content})
        stance = result.strip()
        results.append({"url": url, "stance": stance})

    with open("claim_analysis.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "stance"])
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Results saved to claim_analysis.csv")
    return results

# scipt
if __name__ == "__main__":
    input_claim = input("Enter the claim to verify: ")
    process_claim(input_claim)
