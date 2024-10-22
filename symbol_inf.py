from ultralytics import YOLO

def get_obb_results(image_path: str):
    """
    Perform inference on an image and extract oriented bounding box (OBB) results.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of dictionaries containing class, confidence, and OBB coordinates.
    """
    # Load the trained model
    model = YOLO("best.pt")  # Model path is hardcoded

    # Perform inference
    results = model.predict(source= image_path, save=True)

    # Initialize a list to store the formatted results
    obb_results = []

    # Iterate over the results and format the output
    for result in results:
        if result.obb:
            # Extract OBB information
            obb_data = result.obb
            
            # Iterate over the detected objects
            for i in range(len(obb_data.xywhr)):
                class_id = int(obb_data.cls[i])  # Class ID
                confidence = obb_data.conf[i].item()  # Confidence score
                
                # Only consider results with confidence greater than 0.4
                if confidence > 0.6:
                    x_center, y_center, width, height, rotation = obb_data.xywhr[i]  # OBB coordinates
                    
                    # Append formatted result to the list
                    obb_results.append({
                        "class": result.names[class_id],
                        "confidence": confidence,
                        "obb": {
                            "x_center": x_center.item(),  # Convert tensor to float
                            "y_center": y_center.item(),
                            "width": width.item(),
                            "height": height.item(),
                            "rotation": rotation.item()
                        }
                    })
    print(obb_results)
    return obb_results

# Example usage
if __name__ == "__main__":
    image_file = "test23.jpg"  # Replace with your image path
    results = get_obb_results(image_file)
    #print(results)  # Print the results for debugging
