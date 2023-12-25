from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os


pics_folder = os.path.join('static', 'pics')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = pics_folder 
 
 
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize a dictionary to store the object counts
object_counts = {}

@app.route('/')
def index():
    backgroud = os.path.join(app.config['UPLOAD_FOLDER'], 'detect-obj.png')
    first = os.path.join(app.config['UPLOAD_FOLDER'], 'img-selection.png')
    second = os.path.join(app.config['UPLOAD_FOLDER'], 'Select-img.jpg')
    third = os.path.join(app.config['UPLOAD_FOLDER'], 'Object pic-1.png')

    # Render the HTML form
    return render_template('index.html',bg = backgroud, first_pic = first, second_pic = second ,third_pic = third)

@app.route('/extract')
def detect():
    return render_template('detect.html')

@app.route('/extract', methods=['POST'])
def detect_objects():
    try:
        # Get the uploaded image from the request
        image_file = request.files['image']
        if not image_file:
            return jsonify({"error": "No image provided"})

        # Read the image and perform object detection
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        height, width, channels = image.shape

        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        
        for i in range(len(boxes)):
            if i in indexes:
                label = str(classes[class_ids[i]])
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1
            
       
        response_data = {"object_counts": object_counts}

        return jsonify({"objects": response_data})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

