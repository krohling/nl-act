import torch
from sentence_transformers import SentenceTransformer
import csv
import argparse

def generate_embeddings(input_files, task_ids, train_output_file, val_output_file, train_size=25):
    # Load the MPNet model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    with open(train_output_file, 'w', newline='') as train_csvfile, open(val_output_file, 'w', newline='') as val_csvfile:
        train_writer = csv.writer(train_csvfile)
        val_writer = csv.writer(val_csvfile)
        # Writing header rows
        train_writer.writerow(['task_id', 'instruction', 'embedding'])
        val_writer.writerow(['task_id', 'instruction', 'embedding'])

        for input_file, task_id in zip(input_files, task_ids):
            # Read instructions from the file
            with open(input_file, 'r') as file:
                instructions = [line.strip() for line in file.readlines()]

            # Ensure there are enough instructions for the desired split
            assert len(instructions) >= train_size + 10, "Not enough instructions in the file for the required split"

            # Generate embeddings
            embeddings = model.encode(instructions)

            # Split the data into training and validation sets
            train_instructions = instructions[:train_size]
            val_instructions = instructions[train_size:train_size + 10]
            train_embeddings = embeddings[:train_size]
            val_embeddings = embeddings[train_size:train_size + 10]

            print(f"Task {task_id}: {len(train_instructions)} training instructions, {len(val_instructions)} validation instructions")

            # Save training embeddings and texts to a file
            for instruction, embedding in zip(train_instructions, train_embeddings):
                train_writer.writerow([task_id, instruction, embedding.tolist()])  # Convert tensor to list for saving

            # Save validation embeddings and texts to a file
            for instruction, embedding in zip(val_instructions, val_embeddings):
                val_writer.writerow([task_id, instruction, embedding.tolist()])  # Convert tensor to list for saving

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate embeddings from text instructions across multiple tasks.')
    parser.add_argument('--input_files', nargs='+', type=str, help='Paths to the input text files containing instructions.')
    parser.add_argument('--task_ids', nargs='+', type=int, help='Task IDs corresponding to each input file.')
    parser.add_argument('--train_output_file', type=str, help='Path to save the training output CSV file.')
    parser.add_argument('--val_output_file', type=str, help='Path to save the validation output CSV file.')
    args = parser.parse_args()

    if len(args.input_files) != len(args.task_ids):
        raise ValueError("The number of input files must match the number of task IDs provided.")

    generate_embeddings(args.input_files, args.task_ids, args.train_output_file, args.val_output_file)
