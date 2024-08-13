import numpy as np
import sys   
sys.path.append("..")
import os, glob
import argparse
from torch.backends import cudnn
import torch
from stroke.model import Encoder, Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from stroke.utils import DSC, mkdir
import SimpleITK as sitk
from typing import List, Optional
import pydicom
import nibabel as nib
import scipy as sp
from rt_utils import RTStructBuilder
import scipy as sp
from torch.cuda.amp import autocast
from shapely.geometry import Polygon
from dicompylercore import dicomparser

"""一些相关的函数"""

COLORS = [
    [255, 0, 0],     # Rouge
    [0, 255, 0],     # Vert
    [0, 0, 255],     # Bleu
    [255, 255, 0],   # Jaune
    [0, 255, 255],   # Cyan
    [255, 0, 255],   # Magenta
    [192, 192, 192], # Gris
    [128, 0, 0],     # Bordeaux
    [128, 128, 0],   # Olive
    [0, 128, 0],     # Vert foncé
    [128, 0, 128],   # Pourpre
    [0, 128, 128],   # Sarcelle
    [0, 0, 128],     # Bleu marine
    [255, 165, 0],   # Orange
    [255, 20, 147],  # Rose vif
    [75, 0, 130],    # Indigo
    [255, 192, 203], # Rose pâle
    [70, 130, 180],  # Bleu acier
    [240, 230, 140], # Kaki
    [95, 158, 160]   # Vert cadet
]



#从nii数据中读取data并进行z-score
def get_data_from_nii(raw_data_pth):
    
    # 多模态数据需要标准化
    if 'mask' not in raw_data_pth:
        raw_data = sitk.ReadImage(raw_data_pth)
        data_array = sitk.GetArrayFromImage(raw_data)
        #Origin大于零，表示物理空间中病人头朝下. 多模态计算时需要翻转让其统一头朝上
        if raw_data.GetOrigin()[1]>0:
            data_array = np.flip(data_array,axis=1)
        #进行z-score
        data_mean = np.mean(data_array)
        data_std = np.std(data_array)
        data_array = (data_array - data_mean) / data_std
        
    #mask不需要标准化, 直接取出即可 
    if 'mask' in raw_data_pth:
        raw_data = sitk.ReadImage(raw_data_pth)
        data_array = sitk.GetArrayFromImage(raw_data)
        #Origin大于零，表示物理空间中病人头朝下. 多模态计算时需要翻转让其统一头朝上
        if raw_data.GetOrigin()[1]>0:
            data_array = np.flip(data_array,axis=1)
            
    return data_array

"""0. 环境配置"""
#设置所用GPU (默认单GPU计算)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#定义评估指标 Dice (用来测试是否分割功能正常走通)
criterion = DSC()

"""1. 初始化模型, 单模态模型为(Symbol = 0 (分割adc), Symbol = 1 (分割dwi), Symbol = 2 (分割flair)) 
   多模态模型为 Symbol = 3 (分割多模态)"""
def Create_Model(Symbol = None):

    #1.1 定义所有模型组件
    adc_encoder = Encoder(1)
    dwi_encoder = Encoder(1)
    flair_encoder = Encoder(1)
    decoder = Decoder(1)
    basefuselayer = BaseFeatureExtraction(dim=512, num_heads=8)
    detailfuselayer = DetailFeatureExtraction(dim=512, num_layers=1)

    #1.2 根据Symbol的值的加载对应情况下的模型组件 
    if Symbol != 3:
        
        encoder = Encoder(1)
        
        if Symbol == 0:
            model_name = 'adc'
        if Symbol == 1:
            model_name = 'dwi'        
        if Symbol == 2:
            model_name = 'flair'
            
        ckpts = r'stroke/model/'+ model_name +'.pkl'
        encoder.load_state_dict(torch.load(ckpts,map_location=torch.device('cpu'))['Encoder'])
        basefuselayer.load_state_dict(torch.load(ckpts,map_location=torch.device('cpu'))['BaseFuseLayer'])
        detailfuselayer.load_state_dict(torch.load(ckpts,map_location=torch.device('cpu'))['DetailFuseLayer'])
        decoder.load_state_dict(torch.load(ckpts,map_location=torch.device('cpu'))['Decoder'])
        print("Load Single-modality Model:", ckpts)
        model = [encoder, basefuselayer, detailfuselayer, decoder]
        return model

    if Symbol == 3:
        ckpts = r'./model/Multi.pkl'
        adc_encoder.load_state_dict(torch.load(ckpts)['adc_Encoder'])
        dwi_encoder.load_state_dict(torch.load(ckpts)['dwi_Encoder'])
        flair_encoder.load_state_dict(torch.load(ckpts)['flair_Encoder'])
        basefuselayer.load_state_dict(torch.load(ckpts)['BaseFuseLayer'])
        detailfuselayer.load_state_dict(torch.load(ckpts)['DetailFuseLayer'])
        decoder.load_state_dict(torch.load(ckpts)['Decoder'])
        print("Load Multi-modality Model:", ckpts)
        model = [adc_encoder, dwi_encoder, flair_encoder, basefuselayer, detailfuselayer, decoder]
        return model
    

"""2. 根据地址读取对应数据(预处理: 1.多模态数据统一到相同Resize, 2.配准多模态数据到统一空间, 3.按volume-level进行z-score)
   根据Symbol_D判断对取哪部分数据做预测(与上面的创建model的设置类似), Symbol=0~2表示取单模态数据, Symbol=3表示取多模态数据
   List: 表示数据地址, 当Symbol=3, 按顺序存入的是adc, dwi, flair的数据"""
def Get_Data(Symbol = None, List = None):
    
    #1.2 根据Symbol的值的读取对应情况下的数据
    if Symbol != 3:
        if Symbol == 0:
            data_name = 'adc'
        if Symbol == 1:
            data_name = 'dwi'        
        if Symbol == 2:
            data_name = 'flair'
        single_path =  List[0]
        single_volume = torch.from_numpy(get_data_from_nii(single_path).astype(np.float32))
        print("-----") #输入图像的尺寸为65，224，224
        #print(single_volume.shape)
        single_volume = torch.unsqueeze(single_volume, dim=1)
        
        return single_volume
    
    if Symbol == 3:
        adc_path =  List[0]
        dwi_path = List[1]
        flair_path = List[2]
        adc_volume = torch.from_numpy(get_data_from_nii(adc_path).astype(np.float32))
        dwi_volume = torch.from_numpy(get_data_from_nii(dwi_path).astype(np.float32))
        flair_volume = torch.from_numpy(get_data_from_nii(flair_path).astype(np.float32))

        #对于多模态数据, 拼接为一个volume处理: 形成slice_num*3(modal_num)*height*width的形式
        adc_volume = torch.unsqueeze(adc_volume, dim=1)
        dwi_volume = torch.unsqueeze(dwi_volume, dim=1)
        flair_volume = torch.unsqueeze(flair_volume, dim=1)
        multi_volume = torch.cat((adc_volume, dwi_volume, flair_volume), dim=1)   

        return multi_volume
    
"""3. 利用对应模型和数据进行inference"""
def Get_Pred(model, data):

    data = np.array(data)  # 转换为 numpy 数组
    data = data.transpose(2, 0, 1) 
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data, dim=1) 
    #设置模型为eval进行测试
    if type(model) == list:
        for ids in range(0,len(model)):
            model[ids] = model[ids]#.cuda()
            model[ids] = model[ids].eval()

    with torch.no_grad():

        #释放当前无关显存
        torch.cuda.empty_cache()
        
        #设置batch_size, 并按照batch_size取出测试数据
        slices_num = data.shape[0]
        batch_size = 10
        batch_start = 0
        for iter in range(0, slices_num, batch_size):

            #取出当前iter下的batch
            if slices_num - iter >= batch_size: 
                x = data[batch_start:batch_start + batch_size]
                batch_start += batch_size
            if slices_num - iter < batch_size: 
                x = data[batch_start:slices_num]
            
            #多模态inference 
            if len(model) == 6: 
                #对应模型组件赋值
                adc_encoder = model[0]
                dwi_encoder = model[1]
                flair_encoder =  model[2]
                basefuselayer = model[3]
                detailfuselayer = model[4]
                decoder = model[5]
                
                #对多模态数据batch进行inference
                ADC_put = x[:,0:1,:,:].cuda()
                DWI_put = x[:,1:2,:,:].cuda()
                FLAIR_put = x[:,2:3,:,:].cuda()

                adc_x5,adc_x4,adc_x3,adc_x2,adc_x1,adc_base,adc_detail = adc_encoder(ADC_put)
                dwi_x5,dwi_x4,dwi_x3,dwi_x2,dwi_x1,dwi_base,dwi_detail = dwi_encoder(DWI_put)
                flair_x5,flair_x4,flair_x3,flair_x2,flair_x1,flair_base,flair_detail = flair_encoder(FLAIR_put)

                fusion_base = adc_base + dwi_base + flair_base
                fusion_detail = adc_detail + dwi_detail + flair_detail
                Fusion_share_feature = basefuselayer(fusion_base)
                Fusion_detail_feature = detailfuselayer(fusion_detail)

                Fusion_x4 = (adc_x4 + dwi_x4 + flair_x4)/3
                Fusion_x3 = (adc_x3 + dwi_x3 + flair_x3)/3
                Fusion_x2 = (adc_x2 + dwi_x2 + flair_x2)/3
                Fusion_x1 = (adc_x1 + dwi_x1 + flair_x1)/3

                pred = decoder(torch.cat((Fusion_share_feature, Fusion_detail_feature), dim=1),Fusion_x4,Fusion_x3,Fusion_x2,Fusion_x1)
            
            #单模态inference 
            if len(model) == 4: 
                encoder = model[0]
                basefuselayer = model[1]
                detailfuselayer = model[2]
                decoder = model[3]
                
                #x = x.cuda()
                print(x.shape)
                #对单模态数据batch进行inference
                x5,x4,x3,x2,x1,base,detail = encoder(x)
                share_feature = basefuselayer(base)
                detail_feature = detailfuselayer(detail)
                pred = decoder(torch.cat((share_feature, detail_feature), dim=1),x4,x3,x2,x1)
                print(pred.shape)
            
            #拼接每个batch下的预测
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            if iter==0:
                output=pred
            else:  
                output=torch.cat((output,pred),dim=0)
    output=np.array(output)
    output = output.transpose(1, 2, 3, 0)          
    return output   

def generate_unique_name(rtstruct, base_name):
    """
    Generate a unique name for an ROI by appending a suffix if the name already exists in the RTStruct.

    Args:
        rtstruct (RTStruct): The RTStruct object where the ROI will be added.
        base_name (str): The base name for the ROI.

    Returns:
        str: A unique name for the ROI.
    """

    """
    existing_names = rtstruct.get_roi_names()
    suffix = 1
    unique_name = base_name
    while unique_name in existing_names:
        unique_name = f"{base_name}_{suffix}"
        suffix += 1
    return unique_name
    """
    existing_names =  rtstruct.get_roi_names()  # Get all existing ROI names
    new_names = set()  # Pour garder les noms générés dans cette session
    suffix = 1

    # Vérifier d'abord si un nom de base conflictuel existe
    conflict = any(name.startswith(base_name) for name in existing_names)

    # Si un conflit existe, ajuster tous les noms
    if conflict:
        for name in existing_names:
            if name.startswith(base_name):
                new_name = f"{base_name}_{suffix}"
                while new_name in existing_names or new_name in new_names:
                    suffix += 1
                    new_name = f"{base_name}_{suffix}"
                new_names.add(new_name)
    else:
        # Aucun conflit, utiliser le nom de base directement
        new_names.add(base_name)

    # Renvoyer le dernier nom unique généré ou le nom de base si aucun conflit n'était présent
    return max(new_names, key=len)

def dicom_to_nifti_in_memory(dicom_datasets: List[pydicom.Dataset]) -> nib.Nifti1Image:
    image_slices = [ds.pixel_array for ds in dicom_datasets]
    volume_3d = np.stack(image_slices, axis=-1)
    affine = np.eye(
        4)  # necessaire pour les algorithmes de traitement d'images médicales, ça permet de savoir comment les voxels sont disposés dans l'espace
    nifti_image = nib.Nifti1Image(volume_3d, affine)
    #output_path='/app/rawdata.nii.gz'
    #nib.save(nifti_image, output_path)
    #nifti_image = nifti_image.get_fdata()  # 提取 numpy 数组
    #nifti_image=torch.from_numpy(nifti_image.astype(np.float32))
    #nifti_image = torch.unsqueeze(nifti_image, dim=1)  #[slices,height,weight]
    return nifti_image


"""
Transformations à appliquer sur les images avant de les passer au modèle de segmentation
"""


def transformation():

    return None

def disapplyTransforms(transform, dicoImage):
    dicoImage = transform.inverse(dicoImage)
    return dicoImage["label"], dicoImage["image"]

def applyTransforms(transform, image):
    # Assurez-vous que l'image est un tensor PyTorch
    data_mean = np.mean(image)
    data_std = np.std(image)
    image = (image - data_mean) / data_std
    image = torch.tensor(image, dtype=torch.float32)
    print("===")
    print(image.shape)

    
    #image = (image / torch.max(image)) * 255
    #image = image.unsqueeze(0)
    #image = torch.unsqueeze(image, dim=1) 
    #print(image.shape)
    data = {"image": image, "label": torch.zeros_like(image)}
    transformed=data
    #transformed = transform(data)
    return transformed

def getLabelOfIRM_from_nifti(nifti_image: nib.Nifti1Image, Symbol:int):
    print("rawdata:")
    print(nifti_image.shape)
    transform = transformation()
    transformed_image = applyTransforms(transform, nifti_image.get_fdata())
    niftis=transformed_image["image"]
    print("+++")
    print(niftis.shape)
    #print(type(transformed_image))
    #print(pathModelFile)
    Model = Create_Model(Symbol)
    # dico_image = {"image": transformed_image, "label": torch.zeros_like(transformed_image)}
    Pred = Get_Pred(Model, niftis) #[1,224,224,63]
    label=Pred
    #label = torch.squeeze(Pred, dim=1)
    #label=Pred.sequeeze(dim=1)

    print(type(label))
    output_path='/app/output.nii'
    if not isinstance(label, sitk.Image):
        labels = sitk.GetImageFromArray(label)
        sitk.WriteImage(labels, output_path)
    #label, imageT = disapplyTransforms(transform, dico_image)
    imageT=nifti_image.get_fdata()

    labeled_array, num_features = sp.ndimage.label(label)
    print("翻转回原始尺寸：")
    unique_labels = np.unique(labeled_array)
    print(unique_labels)
    return nifti_image.get_fdata() / 255, labeled_array, imageT

"""
Permet d'écrire sur un RTStruct via les dicoms, un label
:param dicom_datasets: les images dicoms concernées par le RTStruct
:param label: le label obtenu via le modèle
"""
def process_rtstruct_and_calculate_details(dicom_datasets, label, existing_rtstruct=None, voxel_dimensions=(0.5, 0.5, 1.0)):
    print("shape 1")
    print(label.shape)
    if existing_rtstruct:
        rtstruct = RTStructBuilder.create_from_memory(dicom_datasets, existing_rtstruct)
        isFromCurrentRTStruct = True
    else:
        rtstruct = RTStructBuilder.create_new_from_memory(dicom_datasets)
        isFromCurrentRTStruct = False

    results = []
    labeled_array, num_features = sp.ndimage.label(label)
    print("shape 2")
    objects = sp.ndimage.find_objects(labeled_array)

    for i in range(1, num_features + 1):
        mask = np.where(label[ 0,:, :,:] == i, True, False) 
        region_volume = np.sum(mask) * np.prod(voxel_dimensions)  # Volume in mm³
        bbox_lengths = [extent.stop - extent.start for extent in objects[i-1]]
        diameters = [length * voxel for length, voxel in zip(bbox_lengths, voxel_dimensions)]
        start_slice, end_slice = objects[i-1][2].start, objects[i-1][2].stop  # Z-dimension slices

        # Add ROI to RTStruct
        color = COLORS[(i - 1) % len(COLORS)]
        roi_name = f"GTV_MetIA_{i}"
        if existing_rtstruct:
            roi_name = generate_unique_name(rtstruct, f"GTV_MetIA_{i}")
        print(mask.shape)
        rtstruct.add_roi(mask=mask, color=color, name=roi_name)

        # Save details
        results.append({
            'Region ID': i,
            'Volume (mm³)': region_volume,
            'Diameters (mm)': diameters,
            'Start Slice': start_slice,
            'End Slice': end_slice
        })

    return rtstruct, results, isFromCurrentRTStruct


def generate_rtstruct_segmentation_stroke(dicom_datasets: List[pydicom.dataset.Dataset], Symbol: int,
                                         existing_rtstruct: Optional[pydicom.dataset.Dataset] = None):
    """
    Appel le modèle pour générer un RTStruct

    Args :
        dicom_datasets: les images dicoms
        Symbol : uni-multi modality
        existing_rtstruct : le rtstruct sur lequel on se base (optionnel, on peut ne pas en mettre)

    Returns:
        Dataset, Boolean: Le RTStruct correspondant à la segmentation, Est ce que c'est un RTStruct update ou create (faut il remplacer un précédant RTStruct par celui-ci)
    """
    #获取前端传过来的数据；
    niftis = dicom_to_nifti_in_memory(dicom_datasets) #读取的数据为[224,224,63]
    image, label, imageT = getLabelOfIRM_from_nifti(niftis, Symbol)
    #print(niftis.shape)
    #image, label, imageT = getLabelOfIRM_from_nifti(niftis, pathModelFile)
    #1.1 获得对应的模型
    #Model = Create_Model(Symbol)
    #1.3 模型推理产生结果
    #Pred = Get_Pred(Model, niftis)
    #dice = criterion(Pred, label_volume)#.cuda())
    #print(dice)
    rt_struct, metastases_details, isFromCurrentRTStruct = process_rtstruct_and_calculate_details(dicom_datasets, label, existing_rtstruct)
    print("Tout s'est bien passé on dirait")
    for detail in metastases_details:
        print(detail)
    return rt_struct, isFromCurrentRTStruct



if __name__ == '__main__':
    
    """给出两种情况的测试case():
    根据传入的数据分为两种情况, 1.完整的多模态数据  2.不完整的多模态数据(均转换为单模态处理):"""
    """在这两种情况下, 这个模型功能的docker的入参是Symbol + data_list"""

    #1. 当接收到的数据为完整的多模态情况: Symbol = 3, data_list数量为3:
    #传入的测试样例应当为3个模态数据的路径(例如下面, adc_path, dwi_path,flair_path), (label仅仅在模型接入时为了判断是否执行了正确的推理结果的验证, 真实情况下没有这一项)
    Symbol = 1
    base_root = r'E:/VisualSystem/Front_Back/Test_Intergration_1/data/strokecase0010/'
    adc_path =  base_root + 'strokecase0010_adc.nii.gz'
    dwi_path = base_root + 'strokecase0010_dwi.nii.gz'
    flair_path = base_root + 'strokecase0010_flair.nii.gz'
    #data_list = [adc_path, dwi_path, flair_path]
    data_list = [dwi_path]

    label_path = base_root + 'strokecase0010_mask.nii.gz'
    label_volume = torch.from_numpy(get_data_from_nii(label_path).copy().astype(np.uint8))
    label_volume = torch.unsqueeze(label_volume, dim=1)

    #1.1 获得对应的模型
    Model = Create_Model(Symbol)
    
    #1.2 对传入的多模态数据进行预处理
    Data = Get_Data(Symbol, data_list)

    #1.3 模型推理产生结果
    Pred = Get_Pred(Model, Data)
    print(Pred.shape)
    single_volume = torch.squeeze(Pred, dim=1)  #将预测输出的结果转化为原始的图像尺寸；
    outvis=sitk.GetImageFromArray(single_volume)
    sitk.WriteImage(outvis,'E:/VisualSystem/Front_Back/Test_Intergration_1/pred.nii.gz')
#     按volume-level进行evaluation, 验证是否推理正确
    dice = criterion(Pred, label_volume)#.cuda())
    print(dice)

    #2. 当接收到的数据为不完整的模态情况: Symbol = 1/2/3, data_list数量为1, 为1的意思是均转换为单一模态情况的分割
    # (三种模态, dwi->flair->adc, 对lesion的检测能力依次减弱, 应选择可以获得的模态中的对leision检测能力的最强模态):
    # 例如传入了adc和dwi模态, 我们利用dwi模态进行分割获得到的效果比adc的效果是要更好的. 以此类推. 不完整模态下, 仅利用其中一种对leision更具鉴别性的模态进行分割
    # 具体地: Symbol = 0 (分割adc), Symbol = 1 (分割dwi), Symbol = 2 (分割flair)) 多模态模型为 Symbol = 3 (分割多模态)
    
#     Symbol = 1
#     base_root = r'./data/strokecase0010/'
#     adc_path =  None
#     dwi_path = base_root + 'strokecase0010_dwi.nii.gz'
#     flair_path = base_root + 'strokecase0010_flair.nii.gz'
#     data_list = [dwi_path]

#     label_path = base_root + 'strokecase0010_mask.nii.gz'
#     label_volume = torch.from_numpy(get_data_from_nii(label_path).copy().astype(np.uint8))
#     label_volume = torch.unsqueeze(label_volume, dim=1)

#     #1.1 获得对应的模型
#     Model = Create_Model(Symbol)
    
#     #1.2 对传入的多模态数据进行预处理
#     Data = Get_Data(Symbol, data_list)

#     #1.3 模型推理产生结果
#     Pred = Get_Pred(Model, Data)
    
#     #按volume-level进行evaluation, 验证是否推理正确
#     dice = criterion(Pred, label_volume.cuda())
#     print(dice)

    """Pred 即为对应的volume. 注意, 如果要叠加到原始的多模态数据上, 要注意下的物理空间的变化(因为在预处理后所有数据均保持病人头部方向一致)"""