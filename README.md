# SignLanguage_Detection
A real time Sign Language detector using Tensorflow ObjectDetectionAPI 

# Sign Language Detection using TensorFlow Object Detection API

## Introduction
This is the documentation for the sign language detection program using the TensorFlow Object Detection API. The goal of this program is to detect sign language gestures in real-time using a pre-trained object detection model. The program is trained to recognize four classes: "yes", "no", "I love you", and "hello".

[![Sign Language Detection-v1](https://drive.google.com/file/d/1DPJ-qZoma5uiOMR4qHED_Cp1nHSb42gT/view?usp=sharing)](https://drive.google.com/file/d/1DPJ-qZoma5uiOMR4qHED_Cp1nHSb42gT/view?usp=sharing)


## Dataset and Labeling
1. **Dataset**: A custom dataset of sign language gestures was collected for training the model. The dataset includes images of individuals performing the sign language gestures for each class.

2. **Labeling**: The dataset images were labeled using the LabelImg tool. Each image was annotated by drawing bounding boxes around the hand gesture and assigning the corresponding class label.

3. **Data Split**: The labeled dataset was divided into a training set (80%) and a testing set (20%) using a stratified split. This ensures a balanced distribution of each class in both the training and testing sets.

## Label Map and TFRecord Generation
1. **Label Map**: A label map file was created to map class names to integer labels. In this version, the following label map was used:
   ```
   item {
     id: 1
     name: 'yes'
   }
   item {
     id: 2
     name: 'no'
   }
   item {
     id: 3
     name: 'I love you'
   }
   item {
     id: 4
     name: 'hello'
   }
   ```

2. **TFRecord Generation**: The labeled dataset was converted into TensorFlow's TFRecord format for efficient training. This was achieved by running a script that takes the labeled dataset, reads the images and annotations, and creates TFRecord files for the training and testing sets.

## Model Selection and Training
1. **Pre-trained Model**: The pre-trained model used for this program is "ssd_mobilenet_v2_320x320_coco". It is a lightweight object detection model based on the Single Shot Multibox Detector (SSD) architecture.

2. **Pipeline Configuration**: The pipeline configuration file was updated to match the requirements of the sign language detection task. The changes made include:
   - Setting the batch size to 12.
   - Modifying the number of training steps to 25,000.
   - Specifying the path to the label map file.
   - Configuring the input image size to 320x320.

3. **Training Process**: The model was trained using the TensorFlow Object Detection API. This involved running the training script with the modified pipeline configuration and the generated TFRecord files. During training, the model learns to detect and classify the sign language gestures based on the provided dataset.

## Evaluation and Performance Metrics
1. **Testing the Trained Model**: The trained model was evaluated using the testing set. It was run on the testing images to detect sign language gestures, and the results were compared with the ground truth annotations.

2. **Performance Metrics**: Various performance metrics were computed to assess the model's accuracy and effectiveness. These metrics include:
   - Precision: The proportion of correctly detected sign language gestures out of all detected gestures.
   - Recall: The proportion of correctly detected sign language gestures out of all actual gestures in the dataset.
   - F1-score: The harmonic mean of precision and recall, providing an overall measure of the model's performance.

## Future Enhancements
In future versions of this program, several enhancements can be considered, including:

- Collecting a larger and more diverse dataset to improve the model's generalization capabilities.
- Exploring other pre-trained models to achieve higher detection accuracy or real-time performance.
- Fine-tuning the model using transfer learning techniques on a related dataset to boost its performance
