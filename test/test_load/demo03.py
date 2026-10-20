import json

from langchain_core.documents import Document


def load_doc_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return Document(page_content=data['page_content'], metadata=data['metadata'])

if __name__ == '__main__':
    doc = load_doc_from_json("/data/output/1_3.json")
    print(doc)