import os
import fitz
import nltk
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd 
import pickle 
from sentence_transformers import util 
import torch
import textwrap

def print_wrapped(text,wrap_length=80) : 
    wrapped_text = textwrap.fill(text,wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model: SentenceTransformer, n_resources_to_return: int=5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]


    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str, embeddings: torch.tensor, pages_and_chunks: list[dict], n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["chunks"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")


def prompt_formatter(query, context_items,tokenizer, use_dialogue_template=True):
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    # context = "- " + "\n- ".join([item["chunks"] for item in context_items])
    context = " ".join([item["chunks"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """
        Based on the following context items, please answer the query.
        Context item : 
        {context}
        User query: {query}
        Answer:
        """

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    if(use_dialogue_template == True) :
        # Apply the chat template
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
    else : 
        prompt = tokenizer.apply_chat_template(conversation=base_prompt,
                                            tokenize=False,
                                            add_generation_prompt=True) 
    return prompt


def ask(query,tokenizer,llm,embeddings,pages_and_chunks,embedding_model, temperature=0.7, max_new_tokens=512, format_answer_text=True, return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,model=embedding_model)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU 
        
    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items,tokenizer=tokenizer)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text
    
    return output_text, context_items
