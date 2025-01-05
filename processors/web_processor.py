import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from .document_processor import DocumentProcessor
import requests
import hashlib
import os
from functools import lru_cache
import concurrent.futures
import re

class WebProcessor(DocumentProcessor):
    def __init__(self):
        self.cache_dir = ".web_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session = requests.Session()  # Reuse session
        
    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path for URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.txt")

    @lru_cache(maxsize=100)
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with caching."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content while avoiding boilerplate."""
        # Remove unnecessary elements
        for element in soup.select('script, style, nav, footer, header, aside, .advertisement, .sidebar'):
            element.decompose()

        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            return main_content.get_text(separator=' ')
        return soup.get_text(separator=' ')

    async def _fetch_url_async(self, url: str) -> str:
        """Asynchronously fetch URL content."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                return await response.text()

    def process(self, url: str) -> Dict[str, List[Any]]:
        """Process webpage and return extracted text with metadata."""
        try:
            # Check cache first
            cache_path = self._get_cache_path(url)
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_text = f.read()
                return {
                    'texts': [cached_text],
                    'metadata': [{
                        'source': url,
                        'type': 'web',
                        'cached': True
                    }]
                }

            # Fetch content asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            html_content = loop.run_until_complete(self._fetch_url_async(url))
            loop.close()

            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml')  # Using lxml for faster parsing, baki to ho na rahi
            
            # Process content in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Extract title in parallel
                title_future = executor.submit(lambda: soup.title.string if soup.title else url)
                
                # Extract and clean main content
                content_future = executor.submit(self._extract_main_content, soup)
                
                # Get results
                title = title_future.result()
                content = content_future.result()

            # shiny clean text
            cleaned_text = self._clean_text(content)

            # Cache the results
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            return {
                'texts': [cleaned_text],
                'metadata': [{
                    'source': url,
                    'type': 'web',
                    'title': title,
                    'cached': False
                }]
            }

        except Exception as e:
            error_message = str(e)
            if '403' in error_message:
                return {
                    'texts': ['Access Forbidden: Unable to access this webpage. The website has blocked our request.'],
                    'metadata': [{
                        'source': str(url),
                        'type': 'error',
                        'error': 'forbidden_access',
                        'message': error_message
                    }]
                }
            else:
                print(f"Error processing webpage: {error_message}")
                return {
                    'texts': ['Error processing webpage'],
                    'metadata': [{
                        'source': str(url),
                        'type': 'error',
                        'error': 'general_error',
                        'message': error_message
                    }]
                }

    def __del__(self):
        """Cleanup method."""
        self.session.close()