import pandas as pd

def calculate_accuracy(ground_truth_path, predictions_path):
    """
    Calculate accuracy metrics by comparing ground truth and prediction CSV files
    
    Parameters:
    ground_truth_path (str): Path to ground truth CSV file
    predictions_path (str): Path to predictions CSV file
    
    Returns:
    dict: Dictionary containing overall and class-wise accuracy metrics
    """
    # Read CSV files
    ground_truth = pd.read_csv(ground_truth_path)
    predictions = pd.read_csv(predictions_path)
    
    # Merge dataframes on image_name
    comparison = ground_truth.merge(predictions, 
                                  on='image_name', 
                                  suffixes=('_true', '_pred'))
    
    # Calculate accuracy for each class
    total_samples = len(comparison)
    metrics = {}
    
    # Class-wise accuracy
    for column in ['ref_no', 'lot_no', 'use_by']:
        correct = (comparison[f'{column}_true'] == comparison[f'{column}_pred']).sum()
        accuracy = (correct / total_samples) * 100
        metrics[f'{column}_accuracy'] = round(accuracy, 2)
    
    # Overall accuracy (all classes must match)
    all_correct = (
        (comparison['ref_no_true'] == comparison['ref_no_pred']) &
        (comparison['lot_no_true'] == comparison['lot_no_pred']) &
        (comparison['use_by_true'] == comparison['use_by_pred'])
    ).sum()
    
    metrics['overall_accuracy'] = round((all_correct / total_samples) * 100, 2)
    
    # Add sample counts
    metrics['total_samples'] = total_samples
    metrics['correct_samples'] = all_correct
    
    return metrics

def print_accuracy_report(metrics):
    """
    Print a formatted accuracy report
    
    Parameters:
    metrics (dict): Dictionary containing accuracy metrics
    """
    print("\nAccuracy Report")
    print("=" * 50)
    #print(f"Total samples analyzed: {metrics['total_samples']}")
    #print(f"Completely correct samples: {metrics['correct_samples']}")
    print("\nAccuracy Metrics:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']}%")
    print("\nClass-wise Accuracy:")
    print(f"Reference Number: {metrics['ref_no_accuracy']}%")
    print(f"Lot Number: {metrics['lot_no_accuracy']}%")
    print(f"Use By Date: {metrics['use_by_accuracy']}%")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    ground_truth_file = "ocr_results.csv"
    predictions_file = "ocr_predictions.csv"
    
    metrics = calculate_accuracy(ground_truth_file, predictions_file)
    print_accuracy_report(metrics)