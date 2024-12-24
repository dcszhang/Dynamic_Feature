import subprocess
import os
import shutil
def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")


def move_files_to_preprocessed_folder():
    source_files = [
        "data_Dataset.address_to_index", "data_Dataset.labels", "data_Dataset.shuffled_clean_docs",
        "data_Dataset.test_y", "data_Dataset.test_y_prob", "data_Dataset.tfidf_list",
        "data_Dataset.train_y", "data_Dataset.train_y_prob", "data_Dataset.valid_y",
        "data_Dataset.valid_y_prob", "data_Dataset.y", "data_Dataset.y_prob",
        "dev.tsv", "test.tsv", "train.tsv", "weighted_adjacency_matrix.pkl"
    ]

    destination_folder = "../data/preprocessed/Dataset"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in source_files:
        if os.path.exists(file_name):
            shutil.move(file_name, destination_folder)
            print(f"Moved {file_name} to {destination_folder}")
        else:
            print(f"{file_name} does not exist and will not be moved.")
if __name__ == '__main__':
    for i in range(1, 12):
        script_name = f"dataset{i}.py"
        run_script(script_name)
    run_script("adjust_matrix.py")
    run_script("BERT_text_data.py")
    move_files_to_preprocessed_folder()


