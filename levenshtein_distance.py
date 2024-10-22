import csv
import Levenshtein

def calculate_levenshtein_for_csv(file1, file2, output_file):
    """
    Calculate the Levenshtein distance between the 'ref' column of two CSV files
    and write the results to a new file.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w', newline='') as out_file:
        reader1 = csv.DictReader(f1)
        reader2 = csv.DictReader(f2)

        # Check if both CSVs have a 'ref' column
        if 'ref_no' not in reader1.fieldnames or 'ref_no' not in reader2.fieldnames:
            raise ValueError("Both CSV files must have a 'ref' column.")

        writer = csv.writer(out_file)
        writer.writerow(['ref_file1', 'ref_file2', 'levenshtein_distance'])

        # Iterate through both CSVs and calculate the Levenshtein distance
        for row1, row2 in zip(reader1, reader2):
            ref1 = row1['ref_no']
            ref2 = row2['ref_no']
            distance = Levenshtein.distance(ref1, ref2)
            writer.writerow([ref1, ref2, distance])

    print(f"Levenshtein distances have been written to {output_file}.")

# Example usage
file1 = 'ocr_results.csv'
file2 = 'ocr_pred.csv'
output_file = 'levenshtein_distances.csv'

calculate_levenshtein_for_csv(file1, file2, output_file)
