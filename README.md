# **Heart Disease Prediction and Monitoring System**

This project aims to predict and monitor heart disease in real-time using Machine Learning (ML) and IoT technologies. The system uses sensor data to make predictions and provides alerts for high-risk patients.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Machine Learning (ML) Part](#machine-learning-ml-part)
   - [Data Collection](#data-collection)
   - [Feature Selection](#feature-selection)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
3. [Moving the ML Model to IoT](#moving-the-ml-model-to-iot)
   - [Model Optimization](#model-optimization)
   - [Model Conversion to TFLite](#model-conversion-to-tflite)
   - [Deployment on ESP32C3](#deployment-on-esp32c3)
4. [IoT Device Components](#iot-device-components)
   - [Sensors](#sensors)
   - [Microcontroller](#microcontroller)
   - [Communication Modules](#communication-modules)
   - [Power Management](#power-management)
5. [Real-Time Monitoring Workflow](#real-time-monitoring-workflow)
6. [Challenges and Solutions](#challenges-and-solutions)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

---

## **1. Project Overview**
This project combines Machine Learning and IoT to create a real-time heart disease prediction and monitoring system. The ML model predicts heart disease based on sensor data, and the IoT device collects and processes this data to provide actionable insights.

---

## **2. Machine Learning (ML) Part**

### **Data Collection**
- **Dataset**: The project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
- **Features**: The dataset includes features like age, sex, resting blood pressure, ECG results, and more.

### **Feature Selection**
- **Dropped Features**: Features requiring lab tests or advanced medical equipment were removed (e.g., cholesterol, fasting blood sugar).
- **Kept Features**: IoT-compatible features like age, sex, resting blood pressure, ECG results, and heart rate were retained.

### **Model Training**
- **Algorithm**: AdaBoostClassifier.
- **Hyperparameters**:
  - `n_estimators=60`
  - `learning_rate=1`
  - `random_state=100`
- **Training Process**: The model was trained on 80% of the dataset and evaluated on the remaining 20%.

### **Model Evaluation**
- **Metrics**:
  - Accuracy: 81%
  - F1 Score: 81%
  - Precision: 82%
  - Recall: 81%
- **Confusion Matrix**: Used to visualize true positives, false positives, etc.

---

## **3. Moving the ML Model to IoT**

### **Model Optimization**
- **Quantization**: The model was quantized to int8 format to reduce its size and improve performance on the ESP32C3.
- **Pruning**: Unnecessary neurons were pruned to further reduce the model size.

### **Model Conversion to TFLite**
- The trained model was converted to TensorFlow Lite (TFLite) format for deployment on the ESP32C3.
- Conversion Code:
  ```python
  converter = tf.lite.TFLiteConverter.from_saved_model("./model/IoT/heart_disease_model_savedmodel")
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  tflite_model = converter.convert()
  ```

### **Deployment on ESP32C3**
- The TFLite model was converted to a C array and integrated into the ESP32C3 firmware.
- Inference Code:
  ```cpp
  #include <TensorFlowLite.h>
  #include "heart_disease_model_data.cc"

  const tflite::Model* model = tflite::GetModel(heart_disease_model_data);
  tflite::MicroInterpreter interpreter(model, tensor_arena, kTensorArenaSize, nullptr);

  interpreter.AllocateTensors();
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  float input_data[7] = { /* age, sex, trestbps, restecg, thalach, exang, oldpeak */ };
  for (int i = 0; i < 7; i++) {
      input->data.f[i] = input_data[i];
  }

  interpreter.Invoke();
  float prediction = output->data.f[0];
  ```

---

## **4. IoT Device Components**

### **Sensors**
1. **Blood Pressure Sensor**: Measures resting blood pressure (`trestbps`).
2. **ECG Sensor**: Records resting ECG results (`restecg`).
3. **Heart Rate Monitor**: Tracks maximum heart rate (`thalach`).
4. **Activity Monitor**: Infers exercise-induced angina (`exang`).
5. **Heart Rate Variability Sensor**: Approximates ST depression (`oldpeak`).

### **Microcontroller**
- **Seeed XIAO ESP32C3**: A low-power microcontroller with Wi-Fi/Bluetooth capabilities.

### **Communication Modules**
- **Wi-Fi/Bluetooth**: Sends alerts and data to a smartphone app or cloud server.
- **LCD Display**: Displays real-time predictions and alerts.

### **Power Management**
- **Battery**: A rechargeable battery powers the device.
- **Low-Power Modes**: The ESP32C3 uses sleep modes to conserve power.

---

## **5. Real-Time Monitoring Workflow**
1. **Data Collection**: Sensors collect health data (e.g., blood pressure, ECG, heart rate).
2. **Data Preprocessing**: The ESP32C3 preprocesses the data (e.g., normalization, feature extraction).
3. **Inference**: The preprocessed data is fed into the TFLite model for prediction.
4. **Output**: Predictions are displayed on the LCD or sent to a smartphone app.
5. **Alerts**: High-risk predictions trigger alerts (e.g., buzzer, notification).

---

## **6. Challenges and Solutions**
- **Limited Memory**: Optimized the model using quantization and pruning.
- **Power Consumption**: Used low-power sensors and sleep modes.
- **Data Accuracy**: Regularly calibrated sensors and validated predictions.

---

## **7. Future Enhancements**
- **Cloud Integration**: Store and analyze data in the cloud for long-term monitoring.
- **Multi-Patient Support**: Extend the system to monitor multiple patients simultaneously.
- **Advanced Sensors**: Integrate more advanced sensors for better accuracy.

---

## **8. License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **9. Acknowledgements**
- The project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

## **10. References**
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Seeed XIAO ESP32C3](https://www.seeedstudio.com/Seeed-XIAO-ESP32C3-p-4036.html)

## **11. Contact**
- **Kayongo johnson brian**: [Your Name](https://github.com/kaybrian)# capstone
