#  PIC-2-SPEECH

An AI-powered image captioning system that generates textual descriptions for images using the BLIP (Bootstrapped Language-Image Pretraining) model and converts them into audio in five Indian languages (Kannada, Hindi, Marathi, Telugu, Tamil) using Google Text-to-Speech (gTTS).

---

##  Features

-  Image upload and caption generation using BLIP
-  Converts captions to speech using Google Text-to-Speech (gTTS)
-  Translates captions into 5 Indian languages
-  Audio output available for download/playback
-  Evaluation with BLEU and CIDEr metrics

---

##  Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **AI Models**:
  - BLIP (from Hugging Face Transformers)
  - Google Text-to-Speech (gTTS)
  - Google Translator (for multilingual support)
- **Dataset**: Flickr8K

---

##  Dataset

We used the **Flickr8K** image-captioning dataset, which consists of 8,000 images each with 5 captions.  
📎 Download Link: [Flickr8K Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---


##  How to Run the Project
Clone the repository

git clone https://github.com/monishanaik25/Pic-2-Speech.git
cd Pic-2-Speech
Install dependencies

pip install -r requirements.txt
Download the Flickr8K dataset and place it in the data/ directory (or wherever your script expects).

Run the application

python app.py
Access the app in your browser:

http://localhost:5000
 Evaluation
The model is evaluated using:

BLEU Score
Accuracy
CIDEr Score
(Mentioned in the project report or can be added later)

 Future Improvements
Add support for more languages

Improve speech naturalness with advanced TTS models

Add a mobile-friendly UI

 Author
Monisha Naik
📧 [GitHub](https://github.com/monishanaik25)



