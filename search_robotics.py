import arxiv
import json

def search_arxiv_robotics(max_results=30):
    """
    搜尋 arXiv 上關於 robotics 的論文
    """
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=5
    )
    
    search = arxiv.Search(
        query='cat:cs.RO',  # cs.RO 是 Robotics 分類
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    paper_ids = []
    
    print(f"正在搜尋最新的 {max_results} 篇 robotics 論文...\n")
    
    for i, result in enumerate(client.results(search), 1):
        paper_info = {
            'id': result.entry_id.split('/abs/')[-1],
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'published': str(result.published),
            'pdf_url': result.pdf_url
        }
        papers.append(paper_info)
        paper_ids.append(paper_info['id'])
        
        print(f"[{len(papers)}] {paper_info['title'][:80]}...")
        
        if len(papers) >= max_results:
            break
    
    # 保存為 JSON 檔案
    with open('robotics_papers.json', 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    # 保存論文 ID 列表
    ids_string = ','.join(paper_ids)
    with open('paper_ids.txt', 'w') as f:
        f.write(ids_string)
    
    print(f"\n✅ 成功找到 {len(papers)} 篇論文！")
    print(f"📄 論文資訊已保存到: robotics_papers.json")
    print(f"📋 論文 ID 已保存到: paper_ids.txt")
    
    return papers, paper_ids

if __name__ == "__main__":
    # 搜尋 30 篇論文
    papers, ids = search_arxiv_robotics(max_results=30)
    
    # 顯示前 3 篇的詳細資訊作為範例
    print("\n" + "="*80)
    print("前 3 篇論文的詳細資訊：")
    print("="*80)
    for i, paper in enumerate(papers[:3], 1):
        print(f"\n論文 {i}:")
        print(f"標題: {paper['title']}")
        print(f"作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"ID: {paper['id']}")
        print(f"發布日期: {paper['published'][:10]}")
        print(f"摘要: {paper['abstract'][:200]}...")
