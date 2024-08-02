from flask import request, render_template, jsonify,make_response
from app import app#, db
#from models import *
from pydicom import dcmread
import pydicom
import os
import io
from segmentation import generate_rtstruct_segmentation_unetr, extract_roi_info,dicom_to_binary

model_path = "/app/UNETR_model_best_acc.pth"#os.getenv('MODEL_PATH')

@app.route("/")  # 表示访问根路由的时候就会执行下面这个函数；
def home():
    return jsonify(message="Hello, Docker!")


#从前端根据输入的患者信息对患者进行信息建档；
@app.route('/segmentation_unetr', methods=['POST'])
def segmentation_unetr():
    #data = request.get_json()
    #input
    dicom_files = []
    for key in request.files.keys():
        if key.startswith('dicom_data_'):
            dicom_file = request.files[key]
            dicom_data = pydicom.dcmread(io.BytesIO(dicom_file.read()))
            dicom_files.append(dicom_data)
    # 检查上传的文件是否存在
    rtstruct_file = request.files.get('rtstruct_data')
    if rtstruct_file is not None:
        #logging.info(f"Received rtstruct_data file: {rtstruct_file.filename}")
        rtstruct_data = pydicom.dcmread(io.BytesIO(rtstruct_file.read()))
    else:
        #logging.warning("No rtstruct_data file found in the request.")
        rtstruct_data = None
  
    rtstruct, isFromCurrentRTStruct = generate_rtstruct_segmentation_unetr(dicom_files, model_path, rtstruct_data)
    #print(type(rtstruct))
    buffer = rtstruct.save_to_memory()  # 字节流

    # 布尔值转换为字符串
    is_from_current_rtstruct_str = str(isFromCurrentRTStruct)

    # 创建响应对象来返回二进制数据
    response = make_response(buffer.read())
    response.headers.set('Content-Type', 'application/dicom')
    response.headers.set('Content-Disposition', 'attachment; filename=rtstruct_data.dcm')

    # 添加布尔值到响应头
    response.headers.set('X-IsFromCurrentRTStruct', is_from_current_rtstruct_str)

    return response