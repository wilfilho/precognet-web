import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Carrega o modelo a partir do arquivo .keras
model = load_model('precognet-fx-no-optimizer.keras')

def delete_previous_videos():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    block_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        block_frames.append(frame)
        if len(block_frames) == 25:
            block = np.array(block_frames, dtype=np.float32) / 255.0
            block = np.expand_dims(block, axis=0)
            pred = model.predict(block)
            violence = True if pred[0][0] > 0.5 else False
            predictions.append(violence)
            block_frames = []
    cap.release()
    return predictions

@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    # Se for POST, trata upload ou sugestão
    if request.method == 'POST':
        # Caso seja uma sugestão, o formulário oculto terá o campo "suggestion"
        suggestion = request.form.get('suggestion')
        if suggestion:
            suggestion_folder = 'suggestions'
            file_path = os.path.join(suggestion_folder, suggestion)
            if not os.path.exists(file_path):
                error = "Arquivo não encontrado."
                return render_template('index.html', error=error, video_url=None, violence_list=[])
            predictions = process_video(file_path)
            video_url = url_for('suggested_file', filename=suggestion)
            return render_template('index.html', error=error, video_url=video_url, violence_list=predictions)
        # Trata o upload
        if 'video' in request.files:
            file = request.files['video']
            if file.filename == '':
                error = "Nenhum arquivo selecionado."
                return render_template('index.html', error=error, video_url=None, violence_list=[])
            # Verifica extensão MP4
            if not file.filename.lower().endswith('.mp4'):
                error = "Apenas vídeos no formato MP4 são permitidos."
                return render_template('index.html', error=error, video_url=None, violence_list=[])
            # Verifica tamanho (máximo 5 MB)
            file.seek(0, os.SEEK_END)
            filesize = file.tell()
            file.seek(0)
            if filesize > 5 * 1024 * 1024:
                error = "O arquivo deve ter até 5 MB."
                return render_template('index.html', error=error, video_url=None, violence_list=[])
            delete_previous_videos()
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Verifica a duração (máximo 5 segundos)
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else 0
            if duration > 5:
                os.remove(file_path)
                error = "O vídeo deve ter até 5 segundos de duração."
                return render_template('index.html', error=error, video_url=None, violence_list=[])
            predictions = process_video(file_path)
            video_url = url_for('uploaded_file', filename=filename)
            return render_template('index.html', error=error, video_url=video_url, violence_list=predictions)
        return render_template('index.html', error="Erro no envio do arquivo.", video_url=None, violence_list=[])
    else:
        # GET: estado inicial sem vídeo
        return render_template('index.html', error=error, video_url=None, violence_list=[])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/suggestions/<filename>')
def suggested_file(filename):
    return send_from_directory('suggestions', filename)

if __name__ == '__main__':
    app.run(debug=True)
