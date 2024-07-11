import argparse
import os 
import torch 
from tqdm import tqdm 
import pre_preprocess
import query_processing
import pandas as pd 
import numpy as np 
from sentence_transformers import util 
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG")
    parser.add_argument('--doc_dir', default="docs")
    parser.add_argument('--data_pickle', default=r"text_chunks_and_embeddings.pkl")
    parser.add_argument('--skip_last_pages', type=bool, default=False)
    parser.add_argument('--num_skip_last', type=int, default=0)
    # parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--min_token_len', type=int, default=20)
    parser.add_argument('--skip_start_pages', type=bool, default=False)
    parser.add_argument('--num_skip_start', type=int, default=0)
    parser.add_argument('--create_embeddings', type=bool, required=True)
    parser.add_argument('--embedding_model', default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    parser.add_argument('--llm_model', default="google/gemma-2-9b-it")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device : ",device)


    if args.create_embeddings:
        num_last_skip = args.num_skip_last if args.skip_last_pages else 0
        num_start_skip = args.num_skip_start if args.skip_start_pages else 0

        print("Preprocessing the documents.....")
        all_pages_and_chunks = pre_preprocess.process_pdfs(
            directory_path=args.doc_dir,
            # chunk_size=args.chunk_size,
            min_token_len=args.min_token_len,
            skip_start=num_start_skip,
            skip_last=num_last_skip
        )
        print("Done")

        print("Creating embeddings.....")
        all_pages_and_chunks_with_embeddings = pre_preprocess.generate_embeddings(
            all_pages_and_chunks, model_name=args.embedding_model
        )
        print("Done")

        print("Storing information in a pickle file...")
        pre_preprocess.save_to_pickle(all_pages_and_chunks_with_embeddings, save_path=args.data_pickle)
        print("Done")

    # Load the DataFrame including embeddings using pickle
    print("Loading data from pickle file...")
    text_chunks_and_embedding_df = pd.read_pickle(args.data_pickle)
    print(text_chunks_and_embedding_df["embedding"])
    print("Done")

    # Convert embeddings to torch tensor and send to device
    print("text_chunks_and_embedding_df[embedding] : ", text_chunks_and_embedding_df["embedding"] )
    # embeddings_tensor = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    embeddings_list = text_chunks_and_embedding_df["embedding"].tolist()
    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32).to(device)
    print(f"Embeddings shape: {embeddings_tensor.shape}")

    print(f"Embeddings shape: {embeddings_tensor.shape}")

    loaded_df = pd.DataFrame(text_chunks_and_embedding_df)

    # 'embedding' column should remain as numpy arrays
    loaded_df["embedding"] = loaded_df["embedding"].apply(lambda x: np.array(x))

    pages_and_chunks_list = []
    for index, row in text_chunks_and_embedding_df.iterrows():
        chunk_dict = {
            'page_number': row['page_number'],
            'chunks': row['chunk'],
            'chunk_char_count': row['chunk_char_count'],
            'chunk_word_count': row['chunk_word_count'],
            'chunk_token_count': row['chunk_token_count'],
            'embedding': row['embedding']
        }
        pages_and_chunks_list.append(chunk_dict)



    embedding_model = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    # In case you want to reduce the maximum length:
    embedding_model.max_seq_length = 2048

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    flag = True 

    while(flag) :
        query = input("Enter your query : ")
        # Answer query with context and return context 
        answer, context_items = query_processing.ask(query=query, llm=llm, tokenizer=tokenizer, embeddings=embeddings_tensor,
                                                        pages_and_chunks=pages_and_chunks_list, embeddging_model=embedding_model,
                                                        temperature=0.7,
                                                        max_new_tokens=512,
                                                        return_answer_only=False)
        
        print(f"Answer:\n")
        query_processing.print_wrapped(answer)
        print(f"Context items:")
        print(context_items)

        conti = input("More queries (yes/no) :")

        if(conti == "no") : 
            flag = False 
        elif(conti == "yes") : 
            print("continuing....")
        else : 
            print("invalid input, stopping...")
            flag = False 


