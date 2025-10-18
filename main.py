from flask import Flask, request, send_file
import cv2
import numpy as np
import urllib.request
import tempfile

app = Flask(__name__)

@app.route('/')
def enhance():
    url = request.args.get('url')
    if not url:
        return "❌ Missing ?url= parameter", 400

    # تحميل الصورة
    resp = urllib.request.urlopen(url)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # --- خوارزمية التحسين ---
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.7, blur, -0.7, 0)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    final_lab = cv2.merge((cl, a, b))
    final = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(tmp.name, final)
    return send_file(tmp.name, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
