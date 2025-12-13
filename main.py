from chunkingclass import ( 
    CBaseChunkSplitter, 
    CCharTextSplitter, 
    CRecursiveCharTextSplitter, 
    CTiktokenTextSplitter, 
    CTokenTextSplitter, 
    CSpacyTextSplitter, 
    CSentenceTransformersTokenTextSplitter,
    CNLTKTextSplitter,
    CKONLPTextSplitter,
    CGPT2TokenizerFast,
)
from dotenv import load_env

def run_chunking(splitter: CBaseChunkSplitter, docs: str):
    texts = splitter.create_document(docs)
    return texts



if __name__=='__main__':
    load_env()
    DOCFILEPATH = './data/input.txt'
    with open(DOCFILEPATH, "r", encoding='utf-8') as f:
        docs = f.read()


    splitter = CGPT2TokenizerFast(chunk_size=500,chunk_overlap=100)
    result = run_chunking(splitter,docs)
    print(result[0])



