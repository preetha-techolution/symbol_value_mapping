import cv2
import numpy as np
# import concurrent.futures
# from paddleocr import PaddleOCR
import time
import os
from google.cloud import vision
import math
# import imutils
from PIL import Image
# import paddle
# print(paddle.device.get_device())
# print(paddle.device.is_compiled_with_cuda())

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'firebase_keys/vision_key.json'

client = vision.ImageAnnotatorClient()

class ImageOrientationCorrection:
    def __init__(self):
        # self.ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, device='gpu', )
        pass
    
#     def detect_text_orientation(self, image, flag):
#         """
#         Perform OCR and optionally calculate the portrait/landscape orientation counts.
#         """
#         result = self.ocr.ocr(image)

#         if len(result) == 0:
#             return None

#         p_count, l_count, conf_sum, count = 0, 0, 0, 0
        
#         for line in result:
#             for text_info in line:
#                 bounding_box = text_info[0]
#                 conf = text_info[1][1]
#                 conf_sum += conf
#                 if flag == 1:
#                     # Determine portrait/landscape based on bounding box
#                     top_left = bounding_box[0]
#                     top_right = bounding_box[1]
#                     bottom_left = bounding_box[3]
#                     diff_x = abs(top_right[0] - top_left[0])
#                     diff_y = abs(bottom_left[1] - top_left[1])
#                     result = 1 if diff_x >= diff_y else 0
#                     if result == 1:
#                         p_count += 1
#                     else:
#                         l_count += 1
#                 count += 1
        
#         return (0 if p_count >= l_count else 90, conf_sum / count)
    def compute_center(self, vertices):
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        Cx = sum(x_coords) / 4
        Cy = sum(y_coords) / 4
        return (Cx, Cy)

    def compute_angle(self, vertex, center):
        return math.atan2(vertex[1] - center[1], vertex[0] - center[0])

    def detect_text_orientation(self, contour_image_bytes):
        image = vision.Image(content=contour_image_bytes)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if len(texts) == 0:
            print("No text detected")
            return 0

        p_count = 0
        l_count = 0
        count_0 = 0
        count_90 =0
        count_180 =0
        count_270 =0


        for text in texts[1:]:
            # Process bounding boxes concurrently
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            # print(vertices)
            if vertices[1][0] - vertices[0][0] > 0:
                count_0 += 1
            else:
                count_180 += 1
            if vertices[0][0] - vertices[3][0] > 0:
                count_90 += 1
            else:
                count_270 += 1

            center = self.compute_center(vertices)
            angles = [(v, self.compute_angle(v, center)) for v in vertices]
            sorted_vertices = sorted(angles, key=lambda x: x[1])
            sorted_vertices = [v[0] for v in sorted_vertices]
            # print(sorted_vertices)
            top_left = sorted_vertices[0]
            top_right = sorted_vertices[1]
            bottom_left = sorted_vertices[3]
            
            # print(sorted_vertices)

            diff_x = abs(top_right[0] - top_left[0])
            diff_y = abs(bottom_left[1] - top_left[1])

            if diff_x >= diff_y:
                p_count += 1
            else:
                l_count += 1

        # Determine the final orientation based on counts
        # if p_count >= l_count:
        #     return 0 if texts[1].bounding_poly.vertices[1].x - texts[1].bounding_poly.vertices[0].x > 0 else 180
        # else:
        #     return 90 if texts[1].bounding_poly.vertices[0].x - texts[1].bounding_poly.vertices[3].x > 0 else 270
        # print(p_count, l_count)
        if p_count >= l_count:
            if count_0 > count_180:
                return 0
            else: return 180
        else:
            if count_90 > count_270:
                return 90
            else: return 270
    def measure_angle(self, src):
        """
        Measure the initial angle from the binary mask.
        """
        contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rotated_rect = cv2.minAreaRect(largest_contour)
        angle = rotated_rect[-1]
        if angle < -45:
            angle += 90
        # print("measure agnle", angle)
        return angle

    def resize_image(self, image, max_width=1000, max_height=1000):
        height, width = image.shape[:2]
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            if width > height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        return image_rgb

    # def rotate_and_crop(self, mask, image, angle):
    #     (h, w) = image.shape[:2]
    #     center = (w // 2, h // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
    #     # Compute the new bounding dimensions of the image after rotation
    #     cos = np.abs(M[0, 0])
    #     sin = np.abs(M[0, 1])

    #     # New width and height bounds
    #     new_w = int((h * sin) + (w * cos))
    #     new_h = int((h * cos) + (w * sin))

    #     # Adjust the rotation matrix to take into account the translation
    #     M[0, 2] += (new_w / 2) - center[0]
    #     M[1, 2] += (new_h / 2) - center[1]
    
    #     rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #     rotated_image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #     # cv2.imwrite("rotated_image.png", rotated_image)
    #     # cv2.imwrite("rotated_mask.png", rotated_mask)
    #     contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         return rotated_image
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #     cropped_image = rotated_image[y:y + h, x:x + w]
    #     return cropped_image

    # def rotate_and_crop(self, mask, image, angle):
    #     # Step 1: Rotate the image and mask using imutils.rotate_bound (preserves all of the image)
    #     rotated_image = imutils.rotate_bound(image, angle)
    #     rotated_mask = imutils.rotate_bound(mask, angle)
    #     cv2.imwrite("rotated_image.png", rotated_image)
    #     cv2.imwrite("rotated_mask.png", rotated_mask)
    #     # Step 2: Find contours on the rotated mask
    #     contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         print("no contours")
    #         return rotated_image  # If no contours, return the rotated image without cropping
        
    #     # Step 3: Find the largest contour and compute its bounding box
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #     print(x,y,w,h)
    #     # Step 4: Crop the rotated image to the bounding box of the largest contour
    #     cropped_image = rotated_image[y:y + h, x:x + w]
    #     return cropped_image

    def rotate_and_crop(self, mask, image, angle):
        """
        Rotates the image and mask using Pillow, then crops the rotated image based on the largest contour in the mask.
        
        :param mask: The binary mask (NumPy array).
        :param image: The image to rotate and crop (NumPy array).
        :param angle: The angle by which to rotate the image.
        :return: The cropped, rotated image.
        """
        # Convert OpenCV image (NumPy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Pillow
        pil_mask = Image.fromarray(mask)

        # Rotate the image and mask using Pillow
        rotated_pil_image = pil_image.rotate(angle, expand=True)
        rotated_pil_mask = pil_mask.rotate(angle, expand=True)

        # Convert back to NumPy arrays
        rotated_image = np.array(rotated_pil_image)
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        rotated_mask = np.array(rotated_pil_mask)

        # Find contours on the rotated mask
        contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return rotated_image  # Return the rotated image if no contours are found

        # Find the largest contour and its bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the rotated image based on the largest contour's bounding box
        cropped_image = rotated_image[y:y + h, x:x + w]

        return cropped_image

    def get_final_angle(self, mask, image, count = 0, output_dir="data/output/", save_images= True):
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.imread(image_path)

        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        angle = self.measure_angle(mask)

        cropped_image_0 = self.rotate_and_crop(mask, image, angle)
        # cv2.imwrite("cropped_image_0.png", cropped_image_0)
        resized_image_0 = self.resize_image(cropped_image_0)
        _, image_bytes = cv2.imencode('.jpg', resized_image_0, [cv2.IMWRITE_JPEG_QUALITY, 90])

        res_angle = self.detect_text_orientation(image_bytes.tobytes())
        
        # print("res_angle", res_angle)
        # if save_images:
        #     img_name = mask_path.split('/')[-1]

        if save_images:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(f"{output_dir}", exist_ok=True)
        
        if res_angle ==0:
            # print("came to 0")
            final_angle = angle
            res_image = cropped_image_0
            if save_images:
                # print(f"{output_dir}/{count:05}_{final_angle:.2f}.png")
                cv2.imwrite(f"{output_dir}/{count:05}_{final_angle:.2f}.png", cropped_image_0)
        elif res_angle == 180:
            # print("came to 180")
            final_angle = angle + 180
            cropped_image_180 = self.rotate_and_crop(mask, image, final_angle)
            res_image = cropped_image_180
            if save_images:
                # print(f"{output_dir}/{count:05}_{final_angle:.2f}.png")
                cv2.imwrite(f"{output_dir}/{count:05}_{final_angle:.2f}.png", cropped_image_180)
        elif res_angle == 90:
            # print("came to 90")
            final_angle = angle + 90
            cropped_image_90 = self.rotate_and_crop(mask, image, final_angle)
            res_image = cropped_image_90
            if save_images:
                # print(f"{output_dir}/{count:05}_{final_angle:.2f}.png")
                cv2.imwrite(f"{output_dir}/{count:05}_{final_angle:.2f}.png", cropped_image_90)
        elif res_angle == 270:
            # print("came to 270")
            final_angle = angle + 270
            cropped_image_270 = self.rotate_and_crop(mask, image, final_angle)
            res_image = cropped_image_270
            if save_images:
                # print(f"{output_dir}/{count:05}_{final_angle:.2f}.png")
                cv2.imwrite(f"{output_dir}/{count:05}_{final_angle:.2f}.png", cropped_image_270)

        
        # s_t4 = time.time()
#         final_angle = 0

#         if save_images:
#             img_name = mask_path.split('/')[-1]

#         if res_angle == 0:
#             if conf0 > 0.8:
#                 final_angle = angle
#                 if save_images:
#                     cv2.imwrite(f"{output_dir}/{img_name}_{final_angle}.png", cropped_image_0)
#             else:
#                 final_angle = angle + 180
#                 if save_images:
#                     cropped_image_180 = self.rotate_and_crop(mask, image, final_angle)
#                     cv2.imwrite(f"{output_dir}/{img_name}_{final_angle}.png", cropped_image_180)
                         
#         elif res_angle == 90:
#                 cropped_image_90 = self.rotate_and_crop(mask, image, angle+90)
#                 resized_image_90 = self.resize_image(cropped_image_90)
#                 _, conf90 = self.detect_text_orientation(resized_image_90, 0)

#                 if conf90 > 0.8:
#                     final_angle = angle + 90
#                     if save_images:
#                         cv2.imwrite(f"{output_dir}/{img_name}_{final_angle}.png", cropped_image_90)
#                 else:
#                     final_angle = angle + 270
#                     if save_images:
#                         cropped_image_270 = self.rotate_and_crop(mask, image, final_angle)
#                         cv2.imwrite(f"{output_dir}/{img_name}_{final_angle}.png", cropped_image_270)    

        # print("Multiple OCR and Post processing:", time.time() - s_t4)
        return final_angle, res_image


if __name__ == "__main__": 
    # Main script
    obj = ImageOrientationCorrection()
    mask = "Previous Frame_at2000_mask.png"
    image = "Previous Frame_at2000.png"
    output_dir = "results-roi-google"
    save_images = True

    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}", exist_ok=True)
    count = 0
    #for image in os.listdir(img_dir):
        # print(image)
    if image.endswith(".png"):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(image)
            start_time = time.time()
            print("Processing:", image)
            final_angle, res_image = obj.get_final_angle(mask,img, count, output_dir = output_dir, save_images=save_images)
            print("Final angle:", final_angle)
            end_time = time.time()
            print("Time taken:", end_time - start_time)
            print("``````````````````````````````````````````````````````````````")
            count += 1

