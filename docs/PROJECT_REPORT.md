# Comprehensive Project Report: ImageSense AI

**Project Title:** ImageSense AI: Multimodal Neural Image Captioning using Deep Convolutional Feature Extraction and Attention-Driven Recurrent Neural Networks  
**Team Members:** Student & Peer (Mini-Project Group)  
**Domain:** Deep Learning, Computer Vision (CV), Natural Language Processing (NLP)  
**Keywords:** CNN, RNN, LSTM, MobileNetV2, Bahdanau Attention, Flickr8k, BLEU Score  

---

## 1. Abstract
Automatic image caption generation represents an intersection between Computer Vision and Natural Language Processing. While Convolutional Neural Networks (CNNs) model spatial hierarchies to extract semantic representations from grid-like pixel arrays, Recurrent Neural Networks (RNNs) capture temporal dependencies to generate syntactically sound word sequences. This project presents **ImageSense AI**, a deep encoder-decoder framework incorporating **MobileNetV2** as a visual feature extractor and an **LSTM (Long Short-Term Memory)** network as a sequence decoder, interconnected via **Bahdanau Additive Attention**. Evaluated on the held-out split of the Flickr8k benchmark, the system achieves a **BLEU-1 accuracy of 54.0%** and a **BLEU-4 score of 11.6%**. An interactive web interface with client-side Text-to-Speech (TTS) audio synthesis and real-time visual attention heatmap inspection is implemented using Streamlit.

---

## 2. Problem Statement & Motivation
Generating a natural-language description for an image requires more than detecting individual objects: it necessitates understanding spatial interactions, relationships, and context, and expressing those observations in a grammatically fluent sentence. Traditional rule-based image captioning systems rely on hardcoded object templates, which fail when encountering novel compositions. Neural image captioning models replace rigid templates with end-to-end differentiable neural architectures that learn directly from paired image-caption corpora.

---

## 3. System Architecture & Methodology

```
┌─────────────────┐
│   Input Image   │ (224 × 224 × 3 RGB)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ MobileNetV2 CNN Encoder │ (Pretrained on ImageNet, weights frozen)
└────────┬────────────────┘
         │ Feature map: 7 × 7 × 1280 (49 spatial regions)
         ▼
┌─────────────────────────┐
│ Dense Visual Projection │ Projects 1280-d features to 512-d space
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐ ◄─── Hidden state $h_{t-1}$ from LSTM
│ Additive Attention Core │      Score: $e_{t,i} = v^T \tanh(W_q h_{t-1} + W_k k_i)$
└────────┬────────────────┘      Attention weights: $\alpha_{t,i} = \text{softmax}(e_{t,i})$
         │
         │ Context vector $c_t = \sum \alpha_{t,i} k_i$
         ▼
┌─────────────────────────┐ ◄─── Embedded prefix word token ($E(y_{t-1})$)
│ Vision-Language Fusion  │
└────────┬────────────────┘
         │ Concatenated representation (1024-d)
         ▼
┌─────────────────────────┐
│    LSTM Decoder Cell    │ Retains sequential grammatical context
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Dense Output (Softmax) │ Probability distribution over vocabulary (V = 1,078)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Predicted Word ($y_t$) │ Iterated until `<end>` or max length (34)
└─────────────────────────┘
```

### 3.1 Vision Encoder (MobileNetV2)
- Replaces computationally heavy backbones (e.g., VGG-16 with 138M parameters) with depthwise separable convolutions and inverted residual blocks.
- Retains the unpooled $7 \times 7 \times 1280$ activation volume, yielding 49 distinct spatial regions.

### 3.2 Sequence Decoder (LSTM)
- Employs an LSTM memory cell with gating mechanisms ($f_t, i_t, o_t$) to prevent vanishing gradients during multi-step sequence backpropagation.
- Embedding dimension: 256; Hidden units: 512.

### 3.3 Visual Attention Mechanism
- Dynamically assigns an importance score $\alpha_{t,i}$ to each of the 49 image regions at each timestep $t$.
- Enables the decoder to inspect the relevant image patch corresponding to the specific word being produced.

---

## 4. Dataset & Experimental Setup
- **Dataset:** Flickr8k dataset containing 8,091 images, each annotated with 5 human-written captions.
- **Partitioning Strategy:** Partitioned by unique image names into 80% Training, 10% Validation, and 10% Held-Out Testing to strictly avoid reference caption leakage.
- **Training Protocol:** Teacher forcing with Sparse Categorical Cross-Entropy Loss, Adam Optimizer ($\text{lr} = 10^{-3}$), and early stopping.

---

## 5. Quantitative Results & Evaluation

| Metric | Score | Benchmark Interpretation |
| :--- | :--- | :--- |
| **BLEU-1** | **0.540 (54.0%)** | Unigram precision against 5 reference captions |
| **BLEU-4** | **0.116 (11.6%)** | 4-gram fluency and phrase accuracy |
| **Training Loss** | **2.428** | Cross-entropy error on training split |
| **Validation Loss**| **4.178** | Cross-entropy error on unseen validation split |
| **Vocabulary Size**| **1,078 tokens** | Common entities, descriptors, and action verbs |

---

## 6. Qualitative Analysis & Sample Predictions

| Test Image | Predicted Caption | Human Reference Sample | Assessment |
| :--- | :--- | :--- | :--- |
| `2448270671.jpg` | *"A brown dog is about to catch a yellow ball"* | *"A brown dog is leaping over a fallen tree in the woods."* | Correctly identified subject ("brown dog") and action orientation. |
| `2714703706.jpg` | *"A black dog is about to a small dog"* | *"Two dogs are nuzzling each other nose to nose."* | Correctly recognized interaction between multiple dogs. |
| `3526150930.jpg` | *"A young girl in a pink dress is about her"* | *"A woman and a young girl pose and smile for a photo in front of flowers."* | Identified subject ("young girl in a pink dress"). |

---

## 7. Web Application Features
1. **Interactive Attention Heatmap Visualizer:** Computes real-time attention distributions for each generated word, upsampled via bicubic interpolation and blended over the original image with customizable alpha and colormaps (*Jet, Plasma, Inferno, Viridis*).
2. **Text-to-Speech (TTS):** Browser-native Web Speech Synthesis reads the generated caption aloud.
3. **Pre-bundled Gallery:** Instant access to 5 test images for seamless presentation demonstrations.
4. **Diagnostic CV Suite:** Evaluates image resolution, lighting quality, contrast ratios, and dominant color swatches.

---

## 8. Conclusion & Future Enhancements
ImageSense AI demonstrates how combining spatial CNN representations with temporal LSTM decoders enables effective multimodal captioning.
- **Future Directions:**
  1. Integrating Vision Transformers (ViT) and cross-attention Transformer decoders.
  2. Employing Beam Search decoding with length-penalty normalization.
  3. Scaling training to larger corpora such as MS-COCO (330k images).

