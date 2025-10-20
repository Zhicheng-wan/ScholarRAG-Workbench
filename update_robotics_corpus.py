import requests
from bs4 import BeautifulSoup
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

def download_and_extract(url):
    """下載並提取文章內容"""
    try:
        print(f"  下載中...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 提取標題
        title = soup.find('h1')
        if not title:
            title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # 提取內容
        content = None
        for selector in ['article', 'main', '[class*="post-content"]', '[class*="entry-content"]', '[class*="article-content"]', 'body']:
            content = soup.select_one(selector)
            if content:
                break
        
        # 清理不需要的標籤
        if content:
            for tag in content(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()
        
        # 提取文字
        content_text = content.get_text(separator=' ', strip=True) if content else soup.get_text(separator=' ', strip=True)
        
        # 清理多餘空白
        content_text = ' '.join(content_text.split())
        
        return title_text, content_text
        
    except Exception as e:
        print(f"  ✗ 錯誤: {e}")
        return None, None

def create_doc_id(url, index):
    """生成文檔 ID"""
    domain = urlparse(url).netloc
    return f"blog:{domain}#robotics-{index:03d}"

def update_corpus():
    """更新 corpus.jsonl 和 corpus.stats.json"""
    
    urls_file = Path('data/raw/blogs/urls.txt')
    corpus_file = Path('data/processed/corpus.jsonl')
    stats_file = Path('data/processed/corpus.stats.json')
    
    # 檢查檔案
    if not urls_file.exists():
        print(f"❌ 錯誤: 找不到 {urls_file}")
        return
    
    # 讀取新的 URLs
    with open(urls_file, 'r', encoding='utf-8') as f:
        new_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📋 找到 {len(new_urls)} 個新的 URLs\n")
    
    # 讀取現有的 corpus
    existing_data = []
    existing_urls = set()
    
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_data.append(entry)
                    existing_urls.add(entry.get('url', ''))
        print(f"📚 現有文章數: {len(existing_data)}\n")
    
    # 處理新的 URLs
    new_entries = []
    success_count = 0
    
    print("開始處理新文章...")
    print("=" * 80)
    
    for i, url in enumerate(new_urls, 1):
        print(f"\n[{i}/{len(new_urls)}] {url}")
        
        # 檢查是否已存在
        if url in existing_urls:
            print("  ⏭️  跳過（已存在）")
            continue
        
        # 下載內容
        title, content = download_and_extract(url)
        
        if title and content and len(content) > 100:
            # 計算 token 數（粗略估計：1 token ≈ 4 字元）
            tokens = len(content) // 4
            
            # 計算 SHA256
            sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # 生成 doc_id
            doc_id = create_doc_id(url, len(existing_data) + success_count + 1)
            
            # 建立條目（匹配現有格式）
            entry = {
                'doc_id': doc_id,
                'url': url,
                'anchor': '',
                'type': 'blog',
                'title': title,
                'section': 'Body',
                'text': content,
                'source': f"blog:{urlparse(url).netloc}",
                'published': '',
                'authors': '',
                'tokens': tokens,
                'sha256': sha256
            }
            
            new_entries.append(entry)
            success_count += 1
            
            print(f"  ✅ 成功！")
            print(f"     標題: {title[:60]}...")
            print(f"     字數: {len(content.split()):,} words")
            print(f"     Tokens: {tokens:,}")
        else:
            print(f"  ⚠️  內容太短或提取失敗，跳過")
        
        # 禮貌延遲
        if i < len(new_urls):
            time.sleep(2)
    
    print("\n" + "=" * 80)
    
    # 合併並寫入 corpus.jsonl
    all_data = existing_data + new_entries
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 已更新 {corpus_file}")
    print(f"   原有文章: {len(existing_data)}")
    print(f"   新增文章: {len(new_entries)}")
    print(f"   總計文章: {len(all_data)}")
    
    # 更新統計檔案
    total_tokens = sum(entry.get('tokens', 0) for entry in all_data)
    
    # 統計來源分布
    sources = {}
    for entry in all_data:
        source = entry.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    stats = {
        'total_documents': len(all_data),
        'total_tokens': total_tokens,
        'sources': sources,
        'robotics_articles': len(new_entries),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 已更新 {stats_file}")
    print(f"   總文件數: {stats['total_documents']:,}")
    print(f"   總 Tokens: {stats['total_tokens']:,}")
    print(f"   來源分布: {len(sources)} 個不同來源")
    
    print("\n" + "=" * 80)
    print("🎉 完成！")

if __name__ == '__main__':
    update_corpus()
