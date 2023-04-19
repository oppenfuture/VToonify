from flask import Flask
from flask_cors import CORS

def create_app():
  app = Flask(__name__,
    static_url_path='',
    static_folder='./dist')
  from bp import anime_style_transfer
  app.register_blueprint(anime_style_transfer.bp)

  cors = CORS(app)
  app.config['CORS_HEADERS'] = 'Content-Type'
  return app

app = create_app()

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8001, debug=False)