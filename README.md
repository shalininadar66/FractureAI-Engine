# FractureAI-Engine
FractureAI Engine is an end-to-end, cloud-deployed deep learning application designed to assist medical professionals by automating the localized identification of bone fractures from standard X-ray digital radiograph films.

https://huggingface.co/spaces/shalini66/FractureAI-Engine

# FractureAI Engine 🦴

FractureAI Engine is an end-to-end, cloud-deployed deep learning application designed to assist medical professionals by automating the localized identification of bone fractures from standard X-ray digital radiograph films. 

By integrating modern computer vision architectures with dynamic web framework layers, the system translates raw diagnostic imaging data into real-time, interactive clinical insights accessible from any desktop or mobile device.

🚀 **Live Deployment Link:** [View the Live App on Hugging Face](https://huggingface.co/spaces/shalini66/FractureAI-Engine)

---

## 🔬 Core Architecture & Workflow

The platform operates across three seamlessly integrated layers:

1. **Deep Learning Inference (YOLOv8):** At the core sits a trained **YOLOv8** object detection model. When a user submits an X-ray, the model scans the structural pixels down to localized bounding boxes, classifying fracture presence and subtype matrices (e.g., Comminuted Fractures) with precise confidence ratings.
2. **Cloud Infrastructure Hosting (Hugging Face Docker):** The application is fully containerized using **Docker** and hosted permanently inside a virtual Linux environment on **Hugging Face Spaces**. This setup completely decouples processing from local hardware, ensuring 24/7 global access.
3. **Responsive Front-End UI (Streamlit & Tailwind CSS):** A unified data pipeline built with **Streamlit** accepts user uploads via an intuitive sidebar. It encodes processed image predictions into Base64 format and injects them into a high-fidelity, customized diagnostic framework engineered with **Tailwind CSS** for clean, medical-grade visualizations.

---

## 🛠️ Technical Stack Matrix

| Layer | Component / Technology | Purpose |
| :--- | :--- | :--- |
| **Artificial Intelligence** | `ultralytics (YOLOv8)` | Object detection, bounding box localization, and features extraction |
| **Backend Data Pipeline** | `Python`, `OpenCV (Headless)`, `NumPy` | Image buffer processing, matrix parsing, and base64 string encryption |
| **User Interface Layout** | `Streamlit`, `Tailwind CSS Framework` | Cross-platform web interface, sidebar upload management, responsive HTML grid maps |
| **DevOps / Deployment** | `Docker`, `Hugging Face Spaces` | Microservice containerization and secure global cloud server hosting |

---

## 📂 Project Repository Structure

```text
FractureAI-Engine/
├── src/
│   └── streamlit_app.py    <--- Main application source code
├── .gitignore              <--- Prevents temporary cache folders from syncing
├── README.md               <--- Project documentation & portfolio landing page
└── requirements.txt        <--- Application python library dependencies



🔧 Local Installation & Setup
If you want to run this project locally on your machine, follow these steps:

Clone the repository:

Bash
git clone [https://github.com/shalini66/FractureAI-Engine.git](https://github.com/shalini66/FractureAI-Engine.git)
cd FractureAI-Engine
Install the required dependencies:
Make sure you have Python installed, then run:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run src/streamlit_app.py
💡 Key Project Highlights
Automated MLOps Pipeline: Implements a production-grade file streaming mechanism, allowing the lightweight script repository to provision large-scale tensor models dynamically on system boot.

Clinical Safety Protocols: Engineered explicitly with opencv-python-headless dependencies and customized container properties to bypass standard cloud Linux graphic environment constraints (libGL.so.1) smoothly without compromising execution speed.

Edge Case Analysis: Evaluated across diverse bone structures (long-bone tracking profiles vs. complex joint spaces like elbows), providing a clear foundation for studying false-negative risks and data distribution limits in clinical deep learning setups.

Custom built as a data science final year capstone project.
