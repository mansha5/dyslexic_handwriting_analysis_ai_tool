# LearnBridge

LearnBridge is a Streamlit application that analyzes handwritten text using OCR and improves it using a fine-tuned T5 model. It then compares the text before and after correction and generates simple feedback.  
   
## Motivation

Students with dyslexia often struggle with handwriting clarity and grammatical correctness.  
This project aims to assist in analyzing handwritten input and providing simplified, corrected text along with feedback that is easy for parents to understand.  

## Tech Stack

- Python  
- Streamlit  
- Hugging Face Transformers (T5)  
- PyTorch  
- Tesseract OCR  
- Plotly  


## Features

- Upload handwritten image or paste text  
- OCR extraction using Tesseract  
- Text correction using fine-tuned T5 model  
- Grammar and vocabulary score comparison  
- Parent-friendly feedback summary  


## Model Details

- Base Model: T5 (Text-to-Text Transfer Transformer)  
- Task: Text correction and simplification  
- Training Data: Synthetic dataset combining OCR noise and dyslexic patterns  
- Training Method: Sequence-to-sequence fine-tuning  


## How it works

Image → OCR → Extracted Text → T5 Model → Corrected Text → Scoring → Feedback  



## Model

The trained model is not included due to size limitations.  

   To use the app:  
   - Open Model_dyslexia_ai.ipynb in Google Colab.  
   - Run all cells in the notebook.  
   - After training, the notebook will download a model folder named: text_correction_hybrid.zip  
   - Extract the zip file inside the project folder:  
   project  
   │── text_correction_hybrid/  
   │── app.py  
   │── requirements.txt  

## Repository 
   dyslexic_handwriting_analysis_ai_tool/  
   │── app.py  
   │── text_processing.py  
   │── readability_utils.py  
   │── requirements.txt  
   │── Model_dyslexia_ai.ipynb  
   │── theme.qss  
   │  
   ├── screenshots/  
   │   │── output1.png  
   │   │── output2.png  
   │  
   ├── text_correction_hybrid/      # extracted from zip file downloaded from colab  
   │   │── config.json  
   │   │── generation_config.json  
   │   │── model.safetensors  
   │   │── special_tokens_map.json  
   │   │── spiece.model  
   │   │── tokenizer_config.json  
   │   │── tokenizer.json  
   │   │── training_args.bin  
   │  
   └── output/                      # generated automatically  
      │── processed_text.txt  


   ## Setup

   1. Clone the repository:

      ```
      git clone [Repository](https://github.com/mansha5/dyslexic_handwriting_analysis_ai_tool.git)
      cd Project
      ```

   2. Create virtual environment:

      ```
      python3.11 -m venv myenv
      source myenv/bin/activate
      ```

   3. Install dependencies:

      ```
      pip install -r requirements.txt
      brew install tesseract   # for Mac   
      ```

   4. Run:

      ```
      python -m streamlit run app.py
      ```   

## Demo

### Extracted text and cleaned text  
![Output](screenshots/texts.png)  

### Parent Feedback Summary  
![Output](screenshots/feedback.png)  


## Evaluation

The system evaluates text based on:  

- Grammar Score (based on error reduction)  
- Vocabulary Score (based on lexical diversity)  
- Expression Score (based on sentence complexity)  

These scores are heuristic-based and used for comparison between original and corrected text.

## Limitations

- Model is trained on synthetic data, not real dyslexic handwriting  
- OCR accuracy depends on image quality  
- Scoring is heuristic-based and not absolute  




