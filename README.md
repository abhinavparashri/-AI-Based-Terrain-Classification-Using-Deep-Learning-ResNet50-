# -AI-Based-Terrain-Classification-Using-Deep-Learning-ResNet50-
It is main ly used to detect live capturing with any camera like drone ,mobile and satellite image . Insert in this and then you will se then , It will show disater identify ,it  will also show The Recommendaation Action What u will do in that condition 

**#CODE **

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image

# Load AI model
model = ResNet50(weights="imagenet")

# Terrain / weather keywords
nature_keywords = {
    "Rocky": ["rock","mountain","cliff","alp","ridge","valley"],
    "Water": ["sea","ocean","lake","river","lakeside","waterfall","water"],
    "Forest": ["forest","wood","tree","rainforest","pine"],
    "Grassland": ["meadow","field","pasture","prairie","park","lawn"],
    "Sandy": ["sand","beach","sandbar","seashore"],
    "Desert": ["desert","dune"],
    "Snow": ["snow","ice","glacier"],
    "Storm": ["storm","cloud","thundercloud","tornado","cyclone","hurricane"],
    "Fire": ["fire","volcano","lava","eruption"],
    "Agriculture": ["hay","barn","rapeseed","field","crop","farm"]
}

# Disaster detection
def detect_disaster(detected_nature):
    risks = []

    if "Storm" in detected_nature:
        risks.append("Tornado / Thunderstorm")

    if "Water" in detected_nature:
        risks.append("Flood")

    if "Rocky" in detected_nature and "Water" in detected_nature:
        risks.append("Landslide")

    if "Snow" in detected_nature and "Rocky" in detected_nature:
        risks.append("Avalanche")

    if "Fire" in detected_nature:
        risks.append("Volcanic Eruption")

    return risks


# Safety recommendations
def recommend_actions(risks):

    actions = []

    for r in risks:

        if r == "Tornado / Thunderstorm":
            actions.extend([
                "Seek shelter immediately",
                "Avoid open areas",
                "Monitor weather alerts"
            ])

        if r == "Flood":
            actions.extend([
                "Move to higher ground",
                "Avoid flooded roads"
            ])

        if r == "Landslide":
            actions.extend([
                "Stay away from steep slopes",
                "Evacuate unstable terrain"
            ])

        if r == "Avalanche":
            actions.extend([
                "Avoid snowy mountain slopes",
                "Follow avalanche warnings"
            ])

        if r == "Volcanic Eruption":
            actions.extend([
                "Prepare evacuation routes",
                "Keep emergency supplies ready",
                "Monitor volcanic activity alerts",
                "Avoid valleys and low-lying areas near volcanoes"
            ])

    return actions


def detect_nature(image_path):

    img = Image.open(image_path).convert("RGB")

    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()

    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    results = decode_predictions(preds, top=5)[0]

    labels = []

    print("\nDetected Objects\n")

    for r in results:
        label = r[1].lower()
        conf = r[2]*100
        labels.append(label)
        print(label,"-",round(conf,2),"%")

    # Detect terrain types
    detected_nature = set()

    for nature,keywords in nature_keywords.items():
        for label in labels:
            if any(k in label for k in keywords):
                detected_nature.add(nature)

    print("\nDetected Terrain / Weather\n")
    print(detected_nature)

    # Disaster risk
    risks = detect_disaster(detected_nature)

    print("\nDisaster Risk\n")

    if not risks:
        print("No major disaster detected")
    else:
        for r in risks:
            print(r)

    # Recommended actions
    actions = recommend_actions(risks)

    print("\nRecommended Actions\n")

    if not actions:
        print("No action required")
    else:
        for a in actions:
            print("-",a)


# Run
detect_nature("Today 1.jpeg")
