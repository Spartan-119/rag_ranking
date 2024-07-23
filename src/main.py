# main.py

import os
from data_loader import load_job_descriptions, load_kpi_document
from embedder import Embedder
from ranker import Ranker
import pandas as pd

def main():
    # Load data
    try:
        job_descriptions = load_job_descriptions('../data/Indeed end to end - cleaned jd.csv')
    except ValueError as e:
        print(f"Error: {e}")
        column_index = int(input("Please enter the correct column index for job descriptions (0-based): "))
        job_descriptions = load_job_descriptions('../data/Indeed end to end - cleaned jd.csv', column_index)
    
    kpi_document = load_kpi_document('../data/job_ranking.txt')

    # Create embeddings
    embedder = Embedder()
    job_embeddings = embedder.embed(job_descriptions)
    kpi_embedding = embedder.embed(kpi_document)

    # Rank job descriptions
    ranker = Ranker(job_embeddings)
    ranked_indices = ranker.rank(kpi_embedding)

    # Create ranked list of job descriptions
    ranked_job_descriptions = [job_descriptions[i] for i in ranked_indices]

    # Save results
    output_df = pd.DataFrame(ranked_job_descriptions, columns=['Ranked Job Description'])
    output_path = '../data/ranked_job_descriptions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Ranked job descriptions saved to {output_path}")

if __name__ == "__main__":
    main()