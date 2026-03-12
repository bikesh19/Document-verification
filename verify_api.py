import os
import tempfile
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from kyc_verifier import NepalKYCVerifier

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Initialize Verifier
# Note: model.h5 should be in the same directory as this script
MODEL_PATH = "models/model.h5"
verifier = NepalKYCVerifier(classifier_model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)

@app.route('/verify', methods=['POST'])
def verify():
    print(f"Received verification request: {request.files}")
    if 'image' not in request.files:
        print("Error: No image in request.files")
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    def generate():
        try:
            import json
            import time
            
            yield json.dumps({'status': 'initializing'}) + '\n'
            time.sleep(0.5) # Small delay for UI visibility
            
            # Step 1: Classification
            yield json.dumps({'status': 'classification'}) + '\n'
            # We can't easily break into the verifier, so we call it and manage flow
            # Alternatively, we can call the internal methods if they were public
            # For now, we'll call the whole thing but emit "fake" progress to the frontend
            # to show the sequence, or refactor verifier to be a generator.
            # Let's refactor verifier later if needed, for now let's use the sequence.
            
            result = verifier.verify_license(temp_path, verbose=True)
            
            # Since the verifier runs at once, we'll yield the expected sequence
            # but in a real case we'd want the verifier itself to yield.
            # I will refactor verifier.py to be a generator if I have time, 
            # but for now I'll just yield the final result.
            
            yield json.dumps({'status': 'ocr_text_extraction'}) + '\n'
            yield json.dumps({'status': 'field_extracting_parsing'}) + '\n'
            yield json.dumps({'status': 'verifying'}) + '\n'
            
            yield json.dumps({'result': result}) + '\n'
            
        except Exception as e:
            yield json.dumps({'error': str(e)}) + '\n'
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return Response(generate(), mimetype='application/x-ndjson')

if __name__ == '__main__':
    # Disable reloader to prevent restarts when other files in the project are changed
    # Running on port 5000 by default
    print("\nStarting Flask server on http://127.0.0.1:5000")
    print("Reloader is DISABLED. Restart manually if verify_api.py is changed.\n")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
