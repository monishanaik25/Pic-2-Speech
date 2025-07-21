

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
# from PIL import Image, ImageEnhance, ImageFilter
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from googletrans import Translator
# import traceback
# import torch
# import sqlite3
# import os
# from werkzeug.security import generate_password_hash, check_password_hash
# import gtts
# import tempfile
# import uuid
# import numpy as np
# import cv2

# app = Flask(__name__)
# app.secret_key = "your_secret_key_here"
# translator = Translator() 

# # Database setup
# DB_PATH = "pic2speech.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             name TEXT NOT NULL,
#             email TEXT UNIQUE NOT NULL,
#             password TEXT NOT NULL
#         )
#     ''')
#     conn.commit()
#     conn.close()
#     print("Database initialized successfully.")

# # Initialize DB on app start
# init_db()

# # Create temp directory for audio files
# TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), 'pic2speech_audio')
# os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# # Load Fine-Tuned Model
# MODEL_PATH = "finetuned_blip"  

# try:
#     print("Loading Fine-Tuned BLIP model...")
#     processor = BlipProcessor.from_pretrained(MODEL_PATH)
#     model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
#     model.to("cuda" if torch.cuda.is_available() else "cpu")
#     print("Fine-Tuned Model loaded successfully!")
#     MODEL_LOADED = True
# except Exception as e:
#     print(f"Error loading model: {str(e)}")
#     traceback.print_exc()
#     MODEL_LOADED = False

# def detect_main_content_area(image):
#     """
#     Detect and crop the main content area from a screenshot
#     """
#     try:
#         # Convert PIL image to numpy array
#         img_array = np.array(image)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Apply threshold to get binary image
#         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             return image  # Return original if no contours found
        
#         # Find the largest contour (likely the main window)
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         # Add some padding and ensure we don't go out of bounds
#         padding = 20
#         height, width = img_array.shape[:2]
        
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w = min(width - x, w + 2 * padding)
#         h = min(height - y, h + 2 * padding)
        
#         # Crop the image
#         cropped = image.crop((x, y, x + w, y + h))
        
#         return cropped
        
#     except Exception as e:
#         print(f"Error in content detection: {str(e)}")
#         return image  # Return original image if processing fails

# def smart_crop_screenshot(image):
#     """
#     Smart cropping for screenshots - removes taskbars, browser chrome, etc.
#     """
#     try:
#         width, height = image.size
        
#         # Convert to numpy for analysis
#         img_array = np.array(image)
        
#         # Detect uniform colored areas (like taskbars)
#         # Check top area for browser chrome/title bars
#         top_crop = 0
#         for i in range(min(100, height // 4)):
#             row = img_array[i, :, :]
#             # Check if row has uniform color (likely UI element)
#             if np.std(row) < 30:  # Low variance indicates uniform color
#                 top_crop = i + 1
#             else:
#                 break
        
#         # Check bottom area for taskbars
#         bottom_crop = height
#         for i in range(height - 1, max(height - 100, 3 * height // 4), -1):
#             row = img_array[i, :, :]
#             if np.std(row) < 30:
#                 bottom_crop = i
#             else:
#                 break
        
#         # Check left and right for sidebars
#         left_crop = 0
#         for i in range(min(50, width // 4)):
#             col = img_array[:, i, :]
#             if np.std(col) < 30:
#                 left_crop = i + 1
#             else:
#                 break
        
#         right_crop = width
#         for i in range(width - 1, max(width - 50, 3 * width // 4), -1):
#             col = img_array[:, i, :]
#             if np.std(col) < 30:
#                 right_crop = i
#             else:
#                 break
        
#         # Ensure we have a reasonable crop area
#         if (right_crop - left_crop) < width * 0.3 or (bottom_crop - top_crop) < height * 0.3:
#             # If crop area is too small, use center crop
#             center_x, center_y = width // 2, height // 2
#             crop_width = int(width * 0.8)
#             crop_height = int(height * 0.8)
            
#             left_crop = max(0, center_x - crop_width // 2)
#             right_crop = min(width, center_x + crop_width // 2)
#             top_crop = max(0, center_y - crop_height // 2)
#             bottom_crop = min(height, center_y + crop_height // 2)
        
#         # Crop the image
#         cropped = image.crop((left_crop, top_crop, right_crop, bottom_crop))
        
#         return cropped
        
#     except Exception as e:
#         print(f"Error in smart cropping: {str(e)}")
#         return image

# def preprocess_image(image):
#     """
#     Preprocess image to improve caption quality
#     """
#     try:
#         # First try smart cropping for screenshots
#         cropped = smart_crop_screenshot(image)
        
#         # If the cropped image is too different in size, try content detection
#         original_area = image.size[0] * image.size[1]
#         cropped_area = cropped.size[0] * cropped.size[1]
        
#         if cropped_area < original_area * 0.3:  # If we cropped too much
#             cropped = detect_main_content_area(image)
        
#         # Enhance the image quality
#         # Increase contrast slightly
#         enhancer = ImageEnhance.Contrast(cropped)
#         enhanced = enhancer.enhance(1.2)
        
#         # Increase sharpness slightly
#         enhancer = ImageEnhance.Sharpness(enhanced)
#         enhanced = enhancer.enhance(1.1)
        
#         # Resize if image is too large (keep aspect ratio)
#         max_size = 1024
#         width, height = enhanced.size
#         if width > max_size or height > max_size:
#             if width > height:
#                 new_width = max_size
#                 new_height = int((height * max_size) / width)
#             else:
#                 new_height = max_size
#                 new_width = int((width * max_size) / height)
            
#             enhanced = enhanced.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         return enhanced
        
#     except Exception as e:
#         print(f"Error in image preprocessing: {str(e)}")
#         return image

# @app.route('/')
# def landing():
#     if 'user_email' in session:
#         return redirect(url_for('app_page'))
#     return render_template('login.html')

# @app.route('/app')
# def app_page():
#     if 'user_email' not in session:
#         return redirect(url_for('landing'))
#     return render_template('index.html', model_loaded=MODEL_LOADED)

# @app.route('/login', methods=['POST'])
# def login():
#     email = request.form.get('email')
#     password = request.form.get('password')
    
#     if not email or not password:
#         flash('Please provide both email and password')
#         return redirect(url_for('landing'))
    
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
#     user = cursor.fetchone()
#     conn.close()
    
#     if user and check_password_hash(user[3], password):
#         session['user_email'] = email
#         session['user_name'] = user[1]
#         return redirect(url_for('app_page'))
#     else:
#         flash('Invalid email or password')
#         return redirect(url_for('landing'))

# @app.route('/signup', methods=['POST'])
# def signup():
#     name = request.form.get('name')
#     email = request.form.get('email')
#     password = request.form.get('password')
    
#     if not name or not email or not password:
#         flash('Please fill out all fields')
#         return redirect(url_for('landing'))
    
#     hashed_password = generate_password_hash(password)
    
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
    
#     try:
#         cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
#                       (name, email, hashed_password))
#         conn.commit()
#         flash('Account created successfully! Please log in.')
#     except sqlite3.IntegrityError:
#         flash('Email already exists!')
#     finally:
#         conn.close()
    
#     return redirect(url_for('landing'))

# @app.route('/logout')
# def logout():
#     session.pop('user_email', None)
#     session.pop('user_name', None)
#     return redirect(url_for('landing'))

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         if not MODEL_LOADED:
#             return jsonify({'error': 'Model not loaded. Please check server logs.'})

#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})

#         # Process image
#         image = Image.open(file.stream).convert('RGB')
        
#         # Preprocess image (crop and enhance)
#         processed_image = preprocess_image(image)
        
#         # Generate caption using fine-tuned model
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         inputs = processor(processed_image, return_tensors="pt").to(device)
#         output = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
#         caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        
#         return jsonify({
#             'success': True,
#             'caption': caption,
#             'filename': file.filename
#         })
        
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         traceback.print_exc()
#         return jsonify({'error': str(e)})

# @app.route('/translate', methods=['POST'])
# def translate_caption():
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         language = data.get('language', '')

#         if not text:
#             return jsonify({'error': 'No text provided'})
#         if not language:
#             return jsonify({'error': 'No language selected'})
        
#         translated_text = translator.translate(text, dest=language).text
#         return jsonify({'translated_text': translated_text})
    
#     except Exception as e:
#         print(f"Error translating text: {str(e)}")
#         traceback.print_exc()
#         return jsonify({'error': str(e)})

# @app.route('/text-to-speech', methods=['POST'])
# def text_to_speech():
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         language = data.get('language', '')
        
#         if not text:
#             return jsonify({'error': 'No text provided'})
#         if not language:
#             return jsonify({'error': 'No language selected'})
        
#         # Map language codes to gTTS language codes
#         language_map = {
#             'kn': 'kn',  # Kannada
#             'hi': 'hi',  # Hindi
#             'mr': 'mr',  # Marathi
#             'tl': 'te',  # Telugu (corrected to 'te')
#             'tm': 'ta'   # Tamil (corrected to 'ta')
#         }
        
#         # Get the correct language code for gTTS
#         tts_lang = language_map.get(language, 'en')
        
#         # Generate a unique filename for this audio
#         audio_filename = f"{uuid.uuid4()}.mp3"
#         audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        
#         # Generate the audio file
#         tts = gtts.gTTS(text=text, lang=tts_lang, slow=False)
#         tts.save(audio_path)
        
#         # Return the audio file URL
#         return jsonify({
#             'success': True,
#             'audio_url': f'/get-audio/{audio_filename}'
#         })
        
#     except Exception as e:
#         print(f"Error generating speech: {str(e)}")
#         traceback.print_exc()
#         return jsonify({'error': str(e)})

# @app.route('/get-audio/<filename>')
# def get_audio(filename):
#     try:
#         return send_file(os.path.join(TEMP_AUDIO_DIR, filename), 
#                          mimetype='audio/mpeg')
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)





import os
import uuid
import tempfile
import traceback
import sqlite3
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from werkzeug.security import generate_password_hash, check_password_hash
import gtts

app = Flask(__name__)
app.secret_key = "UC4ppA19C0t7aNB9GKGMkUJgg"
translator = Translator()

# Database setup
DB_PATH = "pic2speech.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()

# TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), 'pic2speech_audio')
TEMP_AUDIO_DIR = r"static/audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

MODEL_PATH = "finetuned_blip"
try:
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    MODEL_LOADED = False


@app.route('/')
def landing():
    if 'user_email' in session:
        return redirect(url_for('app_page'))
    return render_template('login.html')


@app.route('/app')
def app_page():
    if 'user_email' not in session:
        return redirect(url_for('landing'))
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[3], password):
        session['user_email'] = email
        session['user_name'] = user[1]
        return redirect(url_for('app_page'))
    else:
        flash('Invalid email or password')
        return redirect(url_for('landing'))


@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    if not name or not email or not password:
        flash('Please fill out all fields')
        return redirect(url_for('landing'))

    hashed_password = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                       (name, email, hashed_password))
        conn.commit()
        flash('Account created successfully! Please log in.')
    except sqlite3.IntegrityError:
        flash('Email already exists!')
    finally:
        conn.close()

    return redirect(url_for('landing'))


@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_name', None)
    return redirect(url_for('landing'))


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if not MODEL_LOADED:
            return jsonify({'error': 'Model not loaded'})

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        image = Image.open(file.stream).convert('RGB')
        width, height = image.size
        aspect_ratio = width / height

        if aspect_ratio > 1.6:
            crop_width = int(width * 0.6)
            crop_height = int(height * 0.6)
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            image = image.crop((left, top, right, bottom))

        temp_path = os.path.join(tempfile.gettempdir(), f"preview_{uuid.uuid4().hex}.png")
        image.save(temp_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]

        return jsonify({
            'success': True,
            'caption': caption,
            'image_path': f'/preview/{os.path.basename(temp_path)}'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/preview/<filename>')
def serve_preview(filename):
    try:
        return send_file(os.path.join(tempfile.gettempdir(), filename), mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/translate', methods=['POST'])
def translate_caption():
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', '')
        if not text:
            return jsonify({'error': 'No text provided'})
        if not language:
            return jsonify({'error': 'No language selected'})
        translated_text = translator.translate(text, dest=language).text
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', '')

        if not text:
            return jsonify({'error': 'No text provided'})
        if not language:
            return jsonify({'error': 'No language selected'})

        language_map = {'kn': 'kn', 'hi': 'hi', 'mr': 'mr', 'tl': 'te', 'tm': 'ta'}
        tts_lang = language_map.get(language, 'en')

        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        tts = gtts.gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(audio_path)

        return jsonify({
            'success': True,
            'audio_url': f'/get-audio/{audio_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get-audio/<filename>')
def get_audio(filename):
    try:
        return send_file(os.path.join(TEMP_AUDIO_DIR, filename), mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
