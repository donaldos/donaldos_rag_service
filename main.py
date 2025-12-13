from chunkingclass import ( 
    CBaseChunkSplitter, 
    CCharTextSplitter, 
    CRecursiveCharTextSplitter, 
    CTiktokenTextSplitter, 
    CTokenTextSplitter, 
    CSpacyTextSplitter, 
    CSentenceTransformersTokenTextSplitter,
    CNLTKTextSplitter,                          # Not completed
    CKONLPTextSplitter,                         # Not completed     
    CGPT2TokenizerFast,                         # Not completed
    CSemanticTextSplitter,
)
from dotenv import load_dotenv

def run_chunking(splitter: CBaseChunkSplitter, docs: str):
    texts = splitter.create_document(docs)
    return texts



if __name__=='__main__':
    load_dotenv()
    DOCFILEPATH = './data/input.txt'
    with open(DOCFILEPATH, "r", encoding='utf-8') as f:
        docs = f.read()
    
    splitter = CSemanticTextSplitter(chunk_size=500,chunk_overlap=100)
    result = run_chunking(splitter,docs)
    print(type(result))
    print(result[0])
    



