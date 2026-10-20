from langchain_community.document_loaders import PyPDFLoader

pdf_file = r"/data/layout-parser-paper.pdf"
# 一页一页解析，每一页对应一个document对象
loader = PyPDFLoader(file_path=pdf_file)

docs = loader.load()

print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content)
