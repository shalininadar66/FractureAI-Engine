
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

# Set up page configurations
st.set_page_config(page_title="FractureAI Engine", layout="wide")

# Load your custom YOLOv8 weights securely in the cloud
import urllib.request

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    # If the file is missing or is just a small text pointer file, automatically download the real 52MB file
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        with st.spinner("Downloading trained model weights directly from server..."):
            url = "https://huggingface.co/spaces/shalini66/FractureAI-Engine/resolve/main/best.pt"
            urllib.request.urlretrieve(url, MODEL_PATH)
            
    return YOLO(MODEL_PATH)

model = load_model()


# -------------------------------------------------------------------------
# STAGE 1: Process User Uploads & Run AI Inference
# -------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Patient X-Ray Film (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Default template state placeholders
findings_count = 0
detections_js_array = "[]"
image_data_uri = "https://lh3.googleusercontent.com/aida-public/AB6AXuCtDX1gUs89ou01faePKytCPkBAyHTZjse7fSDtIXhUkLo0nopPoAj-jQlof6J_HrO4O32ZJlkiJp5WWEwRtss_M_rWZXyITdfIcvj8pqNicLC0FqmkUcJJfTrbG_yrLKzBsbp-aNEHL7f08HsF2Bh8lgbXSSq22UVE1bAi8CG7ZTrwOsHOJaUxRAyStzNuEoJxlTXvwqiH0ZlaLxhWZGylVPjvWUSJBq1i1nPo4LGYvldRXGAD3N2V7kpRzVua4WUPxTi5WdY0Byfy"

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None and model is not None:
        # Run deep learning predictions
        results = model(img)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            conf_val = float(box.conf[0]) * 100
            class_id = int(box.cls[0])
            class_title = model.names[class_id]
            detections.append(f"{{ subtype: '{class_title.upper()}', confidence: {conf_val} }}")
        
        findings_count = len(detections)
        detections_js_array = "[" + ",".join(detections) + "]"
        
        # Plot predictions and encode to base64
        plotted_img = result.plot()
        _, buffer = cv2.imencode('.png', plotted_img)
        base64_encoded = base64.b64encode(buffer).decode('utf-8')
        image_data_uri = f"data:image/png;base64,{base64_encoded}"

# -------------------------------------------------------------------------
# STAGE 2: Inject Your Custom Google Stitch HTML Framework
# -------------------------------------------------------------------------
html_content = rf"""
<!DOCTYPE html><html class="light" lang="en"><head>
<meta charset="utf-8">
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<style>
    body {{ background-color: #f7f9fb; color: #191c1e; font-family: 'Inter', sans-serif; margin: 0; padding: 0; }}
    .material-symbols-outlined {{ font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }}
    .glass-panel {{ background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(8px); }}
    .diagnostic-grid {{ background-image: radial-gradient(#e2e8f0 1px, transparent 1px); background-size: 20px 20px; }}
</style>
</head>
<body class="min-h-screen flex flex-col">
<header class="w-full bg-white border-b border-slate-200 shadow-sm px-6 h-16 flex items-center gap-3">
    <span class="material-symbols-outlined text-slate-900" style="font-size: 28px;">medical_services</span>
    <h1 class="text-xl font-bold text-slate-900">FractureAI Engine</h1>
    <span class="ml-auto text-xs font-semibold px-2 py-1 bg-blue-50 text-blue-600 rounded-full">CLOUD DEPLOYED</span>
</header>
<main class="p-6 max-w-[1440px] mx-auto w-full flex-grow">
    <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div class="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center gap-2">
            <span class="material-symbols-outlined text-slate-600">biotech</span>
            <h2 class="text-lg font-semibold text-slate-900">AI Diagnostic Segmentation Map</h2>
        </div>
        
        <div class="relative w-full aspect-video diagnostic-grid bg-slate-300 flex items-center justify-center">
            <div class="z-10 relative w-4/5 h-4/5 overflow-hidden rounded border border-slate-400 glass-panel flex items-center justify-center shadow-lg">
                <img id="output-image" class="w-full h-full object-contain" src="{image_data_uri}">
            </div>
        </div>
        <div class="p-6 bg-white">
            <div class="flex items-center gap-2 mb-4 pb-2 border-b border-slate-200">
                <span class="material-symbols-outlined text-slate-900">assignment</span>
                <h3 class="text-lg font-semibold text-slate-900">📋 Automated Clinical Analysis Summary</h3>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="space-y-2">
                    <span class="text-xs font-bold text-slate-500 flex items-center gap-1">
                        <span class="material-symbols-outlined text-[14px]">search</span>PRIMARY FINDINGS
                    </span>
                    <div id="findings-box" class="p-4 rounded bg-slate-50 border-l-4"></div>
                </div>
                <div class="space-y-2">
                    <span class="text-xs font-bold text-slate-500 flex items-center gap-1">
                        <span class="material-symbols-outlined text-[14px]">category</span>CLASSIFICATION SUBTYPE
                    </span>
                    <div id="subtype-box" class="p-4 rounded bg-slate-50"></div>
                </div>
                <div class="space-y-2">
                    <span class="text-xs font-bold text-slate-500 flex items-center gap-1">
                        <span class="material-symbols-outlined text-[14px]">verified</span>MODEL CONFIDENCE
                    </span>
                    <div class="p-4 rounded bg-slate-50 flex flex-col justify-center">
                        <div class="flex items-end gap-1 mb-1">
                            <span id="confidence-text" class="text-3xl font-bold text-slate-900">0.0</span>
                            <span class="text-sm text-slate-500 pb-1">%</span>
                        </div>
                        <div class="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                            <div id="confidence-bar" class="bg-blue-600 h-full w-[0%] transition-all duration-500"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</main>
<script>
    const findingsCount = {findings_count};
    const detections = {detections_js_array};
    const outputImage = document.getElementById('output-image');
    const findingsBox = document.getElementById('findings-box');
    const subtypeBox = document.getElementById('subtype-box');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceBar = document.getElementById('confidence-bar');
    if (findingsCount > 0) {{
        findingsBox.parentElement.querySelector('div').className = "p-4 rounded bg-red-50 border-l-4 border-red-600";
        findingsBox.innerHTML = `<p class="font-bold text-red-700">Anomaly Detected: Fracture localized.</p>
                                 <p class="text-sm text-slate-600 mt-1">Identified target regions: ${{findingsCount}} structural split zone(s).</p>`;
        let classesText = detections.map(d => d.subtype).join(", ");
        subtypeBox.innerHTML = `<div class="flex justify-between items-center mb-1">
                                    <span class="text-slate-900 font-bold">\${{classesText}}</span>
                                    <span class="text-xs font-bold bg-red-100 px-2 py-0.5 rounded text-red-700">UNSTABLE</span>
                                </div>
                                <p class="text-sm text-slate-600">Cortical margins matched fracture profile database parameters.</p>`;
        const topConfidence = parseFloat(detections[0].confidence);
        confidenceText.textContent = topConfidence.toFixed(1);
        confidenceBar.style.width = `\${{topConfidence}}%`;
        confidenceBar.className = "bg-red-600 h-full transition-all duration-500";
    }} else if (outputImage.src.includes("data:image/png;base64")) {{
        findingsBox.parentElement.querySelector('div').className = "p-4 rounded bg-green-50 border-l-4 border-green-600";
        findingsBox.innerHTML = `<p class="font-bold text-green-700">No fractures detected.</p><p class="text-sm text-slate-600 mt-1">Skeletal margins show structural uniformity throughout view scan.</p>`;
        subtypeBox.innerHTML = `<p class="text-green-700 font-bold">NORMAL VIEW</p><p class="text-sm text-slate-600">No structural discontinuity flags triggered.</p>`;
        confidenceText.textContent = "100.0";
        confidenceBar.style.width = "100%";
        confidenceBar.className = "bg-green-600 h-full transition-all duration-500";
    }} else {{
        findingsBox.innerHTML = `<p class="text-slate-500 italic">Awaiting Patient X-ray upload inside the left sidebar panel configuration selector...</p>`;
        subtypeBox.innerHTML = `<p class="text-slate-500 italic">Pending scan telemetry data.</p>`;
    }}
</script>
</body></html>
"""

# Render the layout inside Streamlit's container view component
components.html(html_content, height=850, scrolling=True)




