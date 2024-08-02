from flask import Flask
#from flask_sqlalchemy import SQLAlchemy
#from config import Config
from flask_cors import CORS

# 创建Flask类对应的app对象；
app = Flask(__name__)
#app.config.from_object(Config)
CORS(app)
# 初始化数据库
#db = SQLAlchemy(app)

from routes import *

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)