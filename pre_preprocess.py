import os
import fitz
import nltk
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd 
import pickle 

# Download NLTK data if needed
nltk.download('punkt')

def text_formatter(text: str) -> str:
    clean_txt = text.replace("\n", " ").strip()
    return clean_txt

def chunking(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def process_text_and_chunking(pages_and_texts, chunk_size=10, min_token_len=20):
    for item in tqdm(pages_and_texts):
        text = item['text']
        item["sentences"] = nltk.tokenize.sent_tokenize(text, language='english')
        item['page_sentence_count_nltk'] = len(item['sentences'])

        item["chunks"] = chunking(item['sentences'], chunk_size)
        item['num_chunks'] = len(item["chunks"])

        item["chunk_data"] = []
        for chunk in item["chunks"]:
            chunk_dict = {}
            joined_sentence_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            item["chunk_data"].append(chunk_dict)
    
    # Convert to DataFrame and filter based on min_token_len
    all_chunk_data = [chunk for item in pages_and_texts for chunk in item["chunk_data"]]
    df = pd.DataFrame(all_chunk_data)
    pages_and_chunks_over_threshold = df[df["chunk_token_count"] > min_token_len].to_dict(orient="records")
    
    return pages_and_chunks_over_threshold

def process_pdfs(directory_path, min_token_len=20, skip_start=0,skip_last=0):
    pdf_files = [file for file in os.listdir(directory_path) if file.endswith('.pdf')]
    all_pages_and_chunks = []
    pdf_id = 1
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        
        # num_pages = min(len(doc), 11)  

        num_pages = len(doc)

        if(skip_last > num_pages ) :
            print("Skip last cannot be greater than num_pages in any document. Resetting it to 0")
            skip_last=0

        if(skip_start > num_pages) :
            print("Skip start cannot be greater than num_pages in any document. Resetting it to 0")
            skip_start = 0 



        for page_number, page in tqdm(enumerate(doc)):
            if skip_start <= page_number < num_pages - skip_last:  # Process only the first 11 pages as per your original code
                text = page.get_text()
                text = text_formatter(text)
                pages_and_texts.append({
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_file,
                    "page_number": page_number,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split()),
                    "page_sentence_count_raw": len(text.split(". ")),
                    "page_token_count": len(text) / 4,
                    "text": text
                })
        
        processed_pages = process_text_and_chunking(pages_and_texts)
        all_pages_and_chunks.extend(processed_pages)
        pdf_id += 1

    
    
    return all_pages_and_chunks

def generate_embeddings(pages_and_chunks, model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", batch_size=32):
    model = SentenceTransformer(model_name)
    model.max_seq_length = 2048
    for item in tqdm(pages_and_chunks):
        item["embedding"] = model.encode(item["chunk"], batch_size=batch_size, convert_to_numpy=True)
    return pages_and_chunks

def save_embeddings_to_csv(pages_and_chunks, save_path="text_chunks_and_embeddings_df.csv"):
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(save_path, index=False)
    print(f"Embeddings saved to {save_path}")

def save_to_pickle(data, save_path="data.pkl"):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")



# # Example usage:
# directory_path = '/path/to/your/pdf/directory'
# all_pages_and_chunks = process_pdfs(directory_path)

# Now all_pages_and_chunks contains a list of dictionaries, each representing a page from a PDF with chunk-level information, skipping the last 3 pages.
