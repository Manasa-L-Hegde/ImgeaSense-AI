# ImageSense AI: CNN + RNN Mini-Project Viva & Presentation Guide

## 1. Project Overview
- **Project Title:** ImageSense AI — Multimodal Image Captioning with Visual Attention
- **Objective:** Automatically generate natural-language descriptive captions for input images.
- **Core Deep Learning Components:**
  1. **CNN (Convolutional Neural Network):** Pretrained MobileNetV2 for spatial feature representation.
  2. **Attention Mechanism:** Bahdanau (Additive) Attention dynamically aligning words to visual regions.
  3. **RNN (Recurrent Neural Network):** Long Short-Term Memory (LSTM) network decoding sequences token-by-token.
- **Dataset:** Flickr8k (8,091 images, 40,455 reference captions).

---

## 2. Architecture & Dataflow

```
Input Image (224×224×3)
       │
       ▼
[ MobileNetV2 CNN ] (Feature Extractor, weights frozen)
       │
       ▼
Spatial Feature Map: (7×7×1280) = 49 Spatial Regions
       │
       ▼
[ Dense Projection ] ──► (49, 512)
       │
       ▼
[ Bahdanau Additive Attention ] ◄── Query from LSTM previous hidden state (h_t)
       │
       ▼ Context Vector (512)
[ Vision-Language Concatenation ] ◄── Word Embedding (256) + LSTM state
       │
       ▼
[ LSTM Sequence Decoder ] (512 units)
       │
       ▼
[ Dense Softmax Output Layer ] (Vocabulary size = 1,078)
       │
       ▼
Next Predicted Word Token (Iterated until <end> token or max_length=34)
```

---

## 3. Results Summary Table

| Metric | Measured Score | Explanation |
| :--- | :--- | :--- |
| **BLEU-1** | **54.0%** | Single-word accuracy compared against 5 human reference captions. |
| **BLEU-4** | **11.6%** | 4-gram sentence coherence and fluency. |
| **Training Loss** | **2.428** | Categorical Cross-Entropy on teacher-forced sequences. |
| **Validation Loss**| **4.178** | Held-out validation split loss. |
| **Vocabulary** | **1,078 tokens** | Common objects, actions, and descriptors from Flickr8k. |
| **Image Split** | **80% / 10% / 10%** | Image-level partitioning preventing reference caption leakage. |

---

## 4. Top 10 Viva Questions & Expert Answers

### Q1: Why do we combine CNN and RNN together?
> **Answer:** CNNs excel at processing grid-like spatial data (images) and learning hierarchical visual representations (edges -> textures -> objects). However, CNNs cannot natively model variable-length sequential dependencies. RNNs (specifically LSTMs) specialize in sequential time-series modeling. By connecting the CNN feature extractor to an LSTM decoder, the network bridges the gap between vision and natural language.

### Q2: Why choose MobileNetV2 instead of VGG-16 or ResNet-50?
> **Answer:** Standard CNNs like VGG-16 have over 138 million parameters, making them slow and memory-heavy on CPU machines. MobileNetV2 uses **depthwise separable convolutions** and inverted residuals with linear bottlenecks. This achieves comparable feature representation with only ~3.5 million parameters—making training and inference 10x faster while conserving RAM.

### Q3: Why use an LSTM rather than a standard (vanilla) RNN?
> **Answer:** Standard RNNs suffer from the **vanishing and exploding gradient problem** when backpropagating through long sequences, preventing them from learning context beyond 5–10 steps. LSTMs solve this by introducing an internal **cell state** governed by three multiplicative gates:
> - **Forget Gate ($f_t$):** Discards irrelevant past information.
> - **Input Gate ($i_t$):** Decides what new information to store in the cell state.
> - **Output Gate ($o_t$):** Controls what parts of the cell state to output as the hidden state.

### Q4: How does the Visual Attention mechanism work in this project?
> **Answer:** Instead of pooling the entire image into a single global vector (which loses spatial coordinates), we keep the final 7×7 feature map (49 distinct regions). At each word prediction step, **Bahdanau Additive Attention** takes the current LSTM hidden state as a query and compares it against the 49 visual regions to compute a probability distribution (attention weights). The model then takes a weighted sum of these regions, allowing it to "look" at the specific object (e.g. the dog or the cliff) when speaking that word.

### Q5: What is "Teacher Forcing" during training?
> **Answer:** In sequence generation, if the model makes a mistake early in sentence generation (e.g. predicting "cat" instead of "dog"), subsequent words will compound the error. During training, **teacher forcing** supplies the true ground-truth prefix tokens from the reference caption at each step instead of the model's own prior prediction. This stabilizes training and speeds up convergence.

### Q6: How do you evaluate caption quality? What is BLEU?
> **Answer:** We evaluate using the **BLEU (Bilingual Evaluation Understudy)** score against 5 human reference captions provided in Flickr8k:
> - **BLEU-1:** Measures unigram (individual word) precision.
> - **BLEU-4:** Measures 4-gram precision (sequences of 4 consecutive words), evaluating grammatical coherence and phrase structure.
> - A BLEU penalty is applied if generated captions are abnormally short.

### Q7: Why are images split by image name rather than by caption line?
> **Answer:** Flickr8k has 5 captions per image. If we performed a random split across caption lines, one caption for image `A.jpg` might land in the training set while another caption for the exact same `A.jpg` lands in the test set. This would cause data leakage. We perform an **image-level partition (80% train, 10% val, 10% test)** ensuring the model is tested strictly on unseen images.

### Q8: What are the input and output dimensions of each layer?
> **Answer:**
> 1. Input Image: `(224, 224, 3)`
> 2. MobileNetV2 Output: `(7, 7, 1280)` flattened to `(49, 1280)`
> 3. Visual Projection Dense: `(49, 512)`
> 4. Caption Prefix Tokens: `(34,)`
> 5. Word Embedding: `(34, 256)`
> 6. LSTM Hidden State: `(512,)`
> 7. Attention Query: `(1, 512)`
> 8. Attention Context Vector: `(512,)`
> 9. Fusion Concatenation: `(1024,)`
> 10. Next-Word Dense Logits: `(Vocab Size: 1,078)`

### Q9: What are common failure modes or limitations?
> **Answer:**
> 1. Out-of-vocabulary objects: Objects not in the Flickr8k training set (e.g. specialized laptops, modern vehicles) cannot be named.
> 2. Hallucinations / Generic captions: LSTMs can occasionally repeat frequent phrases like "a dog in the grass" if background textures resemble grass.
> 3. Future Improvements: Replacing LSTM with a Transformer decoder (like ViT + GPT/BERT cross-attention) and using Beam Search decoding instead of greedy argmax.

### Q10: How does the Text-to-Speech (TTS) feature work?
> **Answer:** The application uses client-side Web Speech Synthesis (HTML5 Speech API), converting the generated text string directly into synthesized voice audio without network latency or external API costs.

