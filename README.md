# 🚦 Road Accident Detection using Vision-Language Models

**Computer Vision | Deep Learning | Vision-Language Models | PyTorch**

A Master's thesis project focused on detecting road traffic accidents from fixed surveillance camera videos using **Deep Learning and Vision-Language Models (VLMs)**.

The project adapts and extends the **VadCLIP framework**, based on OpenAI's CLIP architecture, to distinguish between accident and normal traffic scenarios through video analysis.

Developed as part of the Master's Degree in Computer Engineering, specializing in **Artificial Intelligence and Intelligent Robotics**, at the University of Salerno.

## 🚀 Key Features

- **Road Accident Detection:** Binary classification of surveillance videos into Accident and Normal scenarios.
- **Vision-Language Models:** Adaptation of VadCLIP with a CLIP ViT-B/16 visual encoder.
- **Parameter-Efficient Fine-Tuning:** Selective fine-tuning of the text encoder to specialize the model.
- **Soft Prompt Learning:** CoOp-inspired learnable prompts for improved text-video alignment.
- **Temporal Video Analysis:** Aggregation of frame-level anomaly scores into video-level predictions.
- **Experimental Evaluation:** Comparison of architectures and analysis across different surveillance conditions.

## 🧠 Proposed Approach

<img width="945" height="495" alt="image" src="https://github.com/user-attachments/assets/38ecdf19-79b1-475a-bb7d-2871948eae52" />

The project extends the original VadCLIP architecture with:

- A task-adapted text encoder using PEFT.
- Shared learnable context tokens inspired by CoOp.
- Multimodal alignment through VadCLIP's A-Branch.
- A video-level score aggregation module based on a lightweight MLP.

Multiple temporal aggregation strategies were evaluated to improve accident detection performance.

## 🛠️ Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch
- **Vision-Language Models:** CLIP, VadCLIP
- **Architectures:** Vision Transformer (ViT-B/16), MLP
- **Techniques:** PEFT, Soft Prompting, Temporal Aggregation
- **Data Processing:** OpenCV, NumPy, Pandas
- **Evaluation:** Scikit-learn

## 📊 Dataset & Results

A dedicated dataset was assembled from multiple sources, containing **2,556 surveillance videos**:

- **1,661** accident videos
- **895** normal traffic videos
- **1,000** held-out test videos

Accident videos were manually annotated to identify the temporal location of accident events.

### Experimental Performance

| Model Configuration | F1-score |
|---|---|
| Original VadCLIP Baseline | 0.626 |
| Exploratory Video-Level MLP (Linear Softmax) | **0.811** |
| Final Joint Video-Level Architecture | 0.772 |

The experiments demonstrated that video-level aggregation substantially improved accident detection compared with the original baseline.

The evaluation also highlighted challenges in nighttime conditions and accidents occurring far from surveillance cameras.

## 🎯 My Contributions

- Built and curated a dedicated road accident video dataset.
- Created temporal annotations for accident events.
- Adapted VadCLIP through parameter-efficient text encoder fine-tuning.
- Implemented a shared soft prompting strategy.
- Developed and evaluated video-level classification modules.
- Conducted experiments, ablation studies and performance analysis.

## 🎓 Thesis Information

- **Title:** Accident Detection on the Road from Video-Surveillance Cameras
- **Author:** Andrea Rescigno
- **University:** University of Salerno
- **Degree:** Master's Degree in Computer Engineering
- **Curriculum:** Artificial Intelligence and Intelligent Robotics
- **Project Type:** Master's Thesis

*This repository presents an academic research prototype for road accident detection, not a production-deployed surveillance system.*
