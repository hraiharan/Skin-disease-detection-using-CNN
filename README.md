

## Skin Disease Detection using Convolutional Neural Networks (CNN)

This repository contains a project aimed at detecting and classifying various skin diseases using Convolutional Neural Networks (CNNs). Leveraging deep learning, this model processes skin images to identify potential conditions such as melanoma, acne, eczema, and more, offering a non-invasive tool to assist in early diagnosis.

### Key Features
- **Dataset Preprocessing**: Includes image resizing, normalization, and data augmentation to improve model robustness.
- **Model Architecture**: Uses CNN architectures like VGG, ResNet, and MobileNet for high accuracy in image classification.
- **Training and Evaluation**: The model is trained on diverse skin disease datasets (e.g., HAM10000, ISIC Archive) and evaluated for metrics such as accuracy, precision, recall, and F1-score.
- **Deployment Ready**: Easily deployable with TensorFlow or PyTorch, with integration options for web or mobile applications.
  
### Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/skin-disease-detection.git
   ```
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the Model and Evaluate the Model**:
   ```bash
   python skin-disease-classification-model.py
   ```
4. **open api folder and open main-tf-serving**:
   ```bash
   python main-tf-serving.py
   ```
4. **Open the Cmd and go to frontend**:
   go .env.example in frontend and change it .env
   ```bash
   cd frontend
   npm install --from-lock-json
   npm audit fix
   npm start
   ```
### Output:

**Running Frontend** 
 ![image](https://github.com/user-attachments/assets/a82fcd37-642f-4429-935c-d6b0139a7861)

**Running Backend**
![image](https://github.com/user-attachments/assets/f44e574f-956b-4d0c-8e6f-637338e313d1)


![image](https://github.com/user-attachments/assets/3d8d6c14-ce9c-4482-b0e5-95455f27a6a5)


![image](https://github.com/user-attachments/assets/7f90f887-7877-4393-b502-780d92e261a7)


![image](https://github.com/user-attachments/assets/8c3ed2ab-ec05-455d-a9e5-3ca3c93ddb5d)
