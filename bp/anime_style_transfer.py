from pathlib import Path
import json, base64, hashlib
from flask import Blueprint, request
from flask_cors import cross_origin
# from backend.matting.rembg_simplify import get_background_mask
# from backend.generativemodels.inpaint import create_inpaint_pipeline
# from backend.generativemodels.inpaint import inpaint
from style_transfer import create_image_style_transfer_cartoon299_models, image_style_transfer_cartoon299
from PIL import Image
from util import encode_image_to_bytes, decode_received_image_data

bp = Blueprint('changeBg', __name__, url_prefix='/changeBg')

models = create_image_style_transfer_cartoon299_models()
index = 0

@bp.route('', methods=('POST', ))
@cross_origin()
def submit_query():
  image_data = request.files['image'].read()
  image = decode_received_image_data(image_data)[:, :, [2, 1, 0]]  # BGR2RGB
#   index += 1
  new_img = image_style_transfer_cartoon299(image, 'cuda', [120, 120, 120, 120], index, *models)[:, :, [2, 1, 0]] # RGB2BGR
  encoded_image = encode_image_to_bytes('.jpg', new_img)
  return json.dumps({
    'format': 'img/jpeg',
    'image': base64.b64encode(encoded_image).decode('utf-8')
  })
