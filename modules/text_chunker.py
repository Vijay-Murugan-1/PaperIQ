from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def create_chunks(text,chunk_size = 512,overlap = 50):
    tokens= tokenizer.encode(text,
                             add_special_tokens = False,
                             truncation = False)
    chunks = []
    stride = chunk_size - overlap

    for start in range (0,len(tokens),stride):

        chunk_tokens = tokens[start : start + chunk_size]

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens = True)
        
        chunks.append(chunk_text)

    return chunks