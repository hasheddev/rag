import re

def chunk_text(text: str, chunk_size: int, overlap: int):
    if overlap >= chunk_size:
        raise ValueError("chunk_size must be strictly greater than overlap")
    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("chunk_size must be strictly greater than overlap")
        
    stripped_txt = text.strip()
    if not stripped_txt:
        return []
    
    boundary = r"(?<=[.!?])\s+"
    sentences = re.split(boundary, stripped_txt)
    
    if len(sentences) == 1 and not sentences[0].endswith(('.', '!', '?')):
        sentences = [stripped_txt]
    
    stripped_sentences = [sentence.strip() for sentence in sentences]
    processed_sentences = [stc for stc in stripped_sentences if stc]
    
    if not processed_sentences:
        return []

    chunks: list[str] = []
    step = max_chunk_size - overlap
    sentence_length = len(processed_sentences)
    
    for i in range(0, sentence_length, step):
        sentence_chunks = processed_sentences[i:i + max_chunk_size]
        chunks.append(" ".join(sentence_chunks))
        
        if i + max_chunk_size >= sentence_length:
            break
            
    return chunks