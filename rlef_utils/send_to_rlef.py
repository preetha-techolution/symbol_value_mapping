"""
This script contains utility functions to send data to RLEF
"""
# importing libraries
import requests
import random
import string
import traceback
import json
import shutil
import os
import threading
import cv2
import uuid 
import datetime 
import warnings
warnings.filterwarnings("ignore")# from settings import config
# importing modules

RLEF_URL = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"


highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                         'rgba(227,0,255,1)']
how_much_annotations_to_skip = 1  # If 70, we will only use every 70th point


def generate_random_id(digit_count):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(digit_count))



def mask_points_to_format(masks, confidence_scores, labels, is_closed=False):
    formatted_data = []

    for mask_points_list, confidence_score, label in zip(masks, confidence_scores, labels):
        random_id_first = generate_random_id(8)
        vertices = []

        for index, (mask_index, mask_points) in enumerate(enumerate(mask_points_list)):
            # Only using every nth point to reduce the number of points based on how_much_annotations_to_skip
            if mask_index % how_much_annotations_to_skip != 0:
                continue

            x, y = mask_points

            # First vertex gets the initial random_id, others get new ids
            vertex_id = random_id_first if index == 0 else generate_random_id(8)

            vertex = {
                "id": vertex_id,
                "name": vertex_id,
                "x": int(x),
                "y": int(y),
            }
            vertices.append(vertex)

        # Ensure that vertices are created, otherwise skip this mask
        if not vertices:
            print("400")
            continue

        mask_data = {
            "id": random_id_first,
            "name": random_id_first,
            "color": random.choice(highContrastingColors),
            "isClosed": is_closed,
            "vertices": vertices,
            "confidenceScore": int(confidence_score * 100),
            "selectedOptions": [
                {
                    "id": "0",
                    "value": "root"
                },
                {
                    "id": random_id_first,
                    "value": label
                }
            ]
        }

        formatted_data.append(mask_data)

    return formatted_data


def send_to_autoai(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type,
                   filename, imageAnnotations, file_type="image/png"):
    try:

        # Create a copy of the file so that it will not be deleted later
        
        if type(filename) != str:
            new_filename = f"runtimeLog/{unique_id}-{str(uuid.uuid1())}.png"
            cv2.imwrite(new_filename, filename)
            # filename = new_filename
        else:
            new_filename = f"runtimeLog/{unique_id}-{str(uuid.uuid1())}_{os.path.basename(filename)}"
            shutil.copy(filename, new_filename)

        if 0 <= float(confidence_score) <= 1:
            confidence_score = confidence_score * 100

        payload = {'status': status,
                   'csv': csv,
                   'model': model,
                   'label': label,
                   'tag': tag,
                   'confidence_score': confidence_score,
                   'prediction': prediction,
                   'imageAnnotations': imageAnnotations,
                   'model_type': model_type}

        files = [('resource', (new_filename, open(new_filename, 'rb'), file_type))]
        headers = {}
        response = requests.request('POST', RLEF_URL, headers=headers, data=payload, files=files, verify=False)
        os.remove(new_filename)
        if response.status_code == 200:
            print('Successfully sent to AutoAI', end="\r")
            return True
        else:
            print('Error while sending to AutoAI')
            print(response.text)
            print(response.status_code)
            return False

    except Exception as e:
        print('Error while sending data to Auto AI : ', e)
        print(traceback.format_exc())
        os.remove(new_filename)
        return False


def send_image_to_rlef(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type,
                       filename, file_type):


    
    threading.Thread(target=send_to_autoai,
                     args=(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type,
                           filename, "[]", file_type,)).start()

    # os.remove(image_name)




# Testing the send_image_to_rlef function
# send_image_to_rlef("backlog", "csv", "63cf83651194684a3191e9f5", "test_label", "test_tag", 0.5, "predicted", "image", "/home/anandhakrishnan/Pictures/IMG_2762.jpg", "image/jpg")


def send_segmentation_to_rlef(unique_id, status, csv, model, image_label, tag, confidence_score, prediction, model_type,
                              filename, file_type, segmentations, confidence_scores, annotation_labels):
    """
    segmentations : List of list containing tuples of X,Ys
    Example : [[(x1, y1), (x2, y2), ....(xn, yn)], [ ]]

    confidence_scores : A list of confidence scores for each segmentation

    labels : A list of labels for each segmentation

    """
    # Convert the annotations to RLEF format

    image_annotations = mask_points_to_format(segmentations, confidence_scores, annotation_labels, is_closed=True)

    threading.Thread(target=send_to_autoai,
                     args=(unique_id, status, csv, model, image_label, tag, confidence_score, prediction, model_type,
                           filename, json.dumps(image_annotations), file_type,)).start()




def send_to_copilot_detection(image, copilot_id):
    filename = f'runtimeLog/{uuid.uuid1()}.png'
    cv2.imwrite(filename, image)
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/coPilotResource/"
    payload = {
        'coPilot': copilot_id,
        'status': 'active',
        'tag': str(datetime.date.today()),
        'type': 'image',
        'name': "sample_image.png",
        'csv': ''
        # 'imageAnnotations': image_annotations
    }
    files = [('resource', ("sample_image.png", open(filename, 'rb'), 'image/png'))]
    try:
        response = requests.post(url, data=payload, files=files, verify=False)
        if response.status_code == 200:
            return response.json()['_id'], True
        else:
            print(response.status_code, response.text)
            print('Error while sending to AutoAI Copilot')
            return None, False
    except Exception as e:
        print('Error while sending data to Auto AI : ', e)
        return None, False
    
   
# Testing the send_segmentation_to_rlef function
# segmentations = [[(1017, 567), (1701, 615), (1710, 981), (971, 930)], [(54, 568), (1000, 346), (683, 687), (872, 123)]]
# confidence_scores = [0.9, 0.8]
# labels = ["screw","the"]
# send_segmentation_to_rlef(unique_id = str(random.randint(0, 9999999)), status = "backlog", csv = "csv", model  = "66e0489214080fc8ee188b2c", image_label = "test", tag = "tag",
#                           confidence_score = "1.0", prediction = "predicted", model_type = "imageAnnotations",
#                           filename = "/Users/vamshikumar/Desktop/onm/asm-ai/runtimeLog/dis_mask.png",
#                           file_type = "image/png", segmentations = segmentations, confidence_scores = confidence_scores, annotation_labels =labels)
