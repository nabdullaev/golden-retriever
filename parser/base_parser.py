import requests
from bs4 import BeautifulSoup
import re
import spacy
from transformers import pipeline, AutoTokenizer


# Initialize models
nlp = spacy.load('en_core_web_sm')
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')


def search_duckduckgo(query):
    """
    Searches DuckDuckGo with the given query and returns a list of results with titles and links.
    """
    formatted_query = query.replace(' ', '+')
    url = f"https://duckduckgo.com/html/?q={formatted_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Referer": "https://duckduckgo.com/",
        "Accept-Language": "en-US,en;q=0.5"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('a', {'class': 'result__a'}):
            title = result.text.strip()
            link = result['href']
            results.append({'title': title, 'link': link})
        return results
    else:
        print(f"Failed to retrieve search results: {response.status_code}")
        return []


def scrape_page_content(url):
    """
    Retrieves the textual content from a given webpage.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Referer": url,
        "Accept-Language": "en-US,en;q=0.5"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.text for para in paragraphs])
    else:
        print(f"Failed to retrieve page content: {response.status_code}")
        return None


def clean_text(text):
    """
    Cleans the input text by removing unnecessary whitespace and special characters.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()


def summarize_text(text):
    """
    Summarizes the input text using a pre-trained summarization model.
    """
    tokens = tokenizer(text, return_tensors="pt").input_ids
    max_token_length = tokenizer.model_max_length
    input_length = len(tokens[0])
    
    if input_length > max_token_length:
        print(f"Input too long: {input_length} tokens (max: {max_token_length}), truncating...")
        text = tokenizer.decode(tokens[0][:max_token_length], skip_special_tokens=True)
    
    try:
        max_length = min(input_length // 2, 150)
        min_length = min(input_length // 4, 30)
        
        if input_length < 10:
            return text  # Skip summarization if input is too short

        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text


def extract_keywords(text):
    """
    Extracts keywords from a given text using spaCy.
    """
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and not token.is_punct]
    return list(set(keywords))


def main():
    # Sample user query and keyword extraction
    user_query = "Find me restaurants in London where I can go with my family to eat Sushi or other asian cuisine"
    keywords = extract_keywords(user_query)
    clear_query = ' '.join(keywords)
    print(f"Generated Search Query: {clear_query}")

    # Search using the clear query and scrape results
    results = search_duckduckgo(clear_query)

    if results:
        scraped_contents = []
        for result in results:
            content = scrape_page_content(result['link'])
            if content:
                cleaned_content = clean_text(content)
                scraped_contents.append(cleaned_content)
        
        # Summarize the scraped content
        summaries = [summarize_text(content) for content in scraped_contents]

        for i, summary in enumerate(summaries):
            print(f"Summary {i + 1}:\n{summary}\n")
    else:
        print("No results found.")


if __name__ == "__main__":
    main()
