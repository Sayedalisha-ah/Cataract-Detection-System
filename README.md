# 👁️ Cataract Detection using Vision Transformers

This project uses **Vision Transformers (ViTs)** on ophthalmic **slit-lamp images** for **automatic cataract detection and grading**.  
The model focuses on analyzing the lens region and classifies the severity of cortical cataracts, helping in early diagnosis and treatment planning.

---

## 📌 Project Highlights
- Built with **Vision Transformers (ViT)** for advanced image classification.  
- Trained on ophthalmic slit-lamp images graded using the **LOCS III system**.  
- Handles complex cataract patterns by focusing on both **central** and **peripheral lens regions**.  
- Designed for practical **real-world medical use** — requires only images as input.  

---

## 📊 Model Performance (Test Data)
- **Accuracy:** ~95% (high reliability in cataract grading)  
- **Precision:** ~73%  
- **Recall:** ~71%  
- **F1-score:** ~72%  

---

## ⚙️ Tech Stack
- **Framework:** PyTorch  
- **Model:** Vision Transformer (ViT)  
- **Preprocessing:** Faster R-CNN for lens region detection, image resizing to `224x224`  
- **Dataset:** Slit-lamp ophthalmic images labeled with **LOCS III grading system**  

---

## 🚀 Workflow
1. **Image Preprocessing** → Crop lens region, resize, normalize.  
2. **Model Training** → Vision Transformer learns cataract severity from labeled images.  
3. **Prediction** → Input a new eye image → output cataract grade (0–6 based on LOCS III).  

---

## 📂 Repository Structure
