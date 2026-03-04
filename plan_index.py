import os
import pandas as pd
from pydicom import dcmread
import numpy as np
from dicompylercore import dicomparser, dvhcalc

def find_DV_for_OAR(stru_name, dose_centers, volumes):
    metrics = {}
    # --- 计算 D% 指标 (例如 D95, D2) ---
    # D95: 被95%体积所接受的剂量
    target_vols = [95, 100]
    for tv in target_vols:
        target_a_vol = tv / 100 * volumes[0]
        indices = np.where(volumes < target_a_vol)[0]
        if len(indices) > 0:
            idx = indices[0]
            if idx > 0:
                # 线性插值
                v1, v2 = volumes[idx-1], volumes[idx]
                d1, d2 = dose_centers[idx-1], dose_centers[idx]
                interpolated_dose = d1 + (target_a_vol - v1) / (v2 - v1) * (d2 - d1)
                metrics[f"{stru_name}_D{tv}%"] = round(interpolated_dose, 3)
            else:
                # 如果第一个点就满足条件，取该点剂量
                metrics[f"{stru_name}_D{tv}%"] = round(dose_centers[idx], 3)
        else:
            # 如果没有找到，则该结构未达到此体积覆盖率
            metrics[f"{stru_name}_D{tv}%"] = None

    return metrics

def find_VD_for_OAR(stru_name, target_doses_gy, dose_centers, volumes):
    metrics = {}
    # --- 计算 V% 指标 (例如 V100, V20) ---
    # V20Gy: 接受20Gy以上剂量的体积百分比
    for td_gy in target_doses_gy:
        # 在剂量轴上，找剂量 >= td_gy 的第一个点
        indices = np.where(dose_centers >= td_gy)[0]
        if len(indices) > 0:
            # 在累积DVH中，高剂量处的体积低。所以第一个满足条件的点的volume就是V(td_gy)
            # volume的第一个就是总体积
            volume_td = volumes[indices[0]] / volumes[0] * 100
            metrics[f"{stru_name}_V{int(td_gy):0>2d}"] = round(volume_td, 2)
        else:
            # 如果所有剂量都低于 td_gy，则体积覆盖率为0
            metrics[f"{stru_name}_V{int(td_gy):0>2d}"] = 0.0 
    
    return metrics

def find_Dmax_for_OAR(stru_name, dose_centers, volumes):
    metrics = {}
    # 最大剂量 Dmax (Gy) - 对应体积最小的剂量点
    dmax = dose_centers[np.argmin(volumes)]
    metrics[f"{stru_name}_Dmax"] = round(dmax, 3)
    return metrics

def find_Dmin_for_OAR(stru_name, dose_centers, volumes):
    metrics = {}
    # 最小剂量 Dmin (Gy) - 对应体积最大的剂量点
    dmin = dose_centers[np.argmax(volumes)]
    metrics[f"{stru_name}_Dmin"] = round(dmin, 3)
    return metrics

def find_Dmean_for_OAR(stru_name, dose_centers, volumes):
    metrics = {}
    # 平均剂量 Dmean (Gy) - 使用体积加权平均
    # Delta_Vol = difference in cumulative volume
    delta_vols = np.abs(np.diff(np.append(volumes, 0))) # 近似每个bin的体积
    total_vol = volumes[0] # 总体积
    if total_vol > 0:
        weighted_dose_sum = np.sum(dose_centers * delta_vols)
        dmean = weighted_dose_sum / total_vol # 
        metrics[f"{stru_name}_Dmean"] = round(dmean, 3)
    else:
        metrics[f"{stru_name}_Dmean"] = None
    
    return metrics

def calculate_dvh_metrics(dvh_data, structure_name):
    if dvh_data is None:
        return {}
    # 获取累积DVH的剂量和体积数据
    # dose_values 是剂量数组 (通常是 bin 边界)
    # volume_values 是对应的累积体积百分比
    try:
        # 尝试获取累积DVH数据
        dose_values = dvh_data.bins # 这是剂量bins，长度为 N+1
        volume_values = dvh_data.counts # 这是累积体积%，长度为 N
    except AttributeError:
        # 如果 bins 不可用，尝试其他方式
        # 检查对象的所有属性
        print(f"  - 调试信息: DVH 对象属性 -> {dir(dvh_data)}")
        # 通常，dvh_data.dose.data 包含剂量，dvh_data.volume 包含体积
        # 让我们尝试另一种常见结构
        if hasattr(dvh_data, 'dose') and hasattr(dvh_data, 'volume'):
            # 如果是这种结构，需要重新评估如何访问数据
            # 通常情况下，dvhcalc 返回的对象应有 .bins 和 .counts
            # 如果没有，说明对象结构不同
            # 最后尝试获取原始统计信息
            stats = dvh_data.get_stats()
            print(f"  - 结构 {structure_name} 的统计信息: {stats}")
            return {}
        else:
            return {}

    # 确保我们有数据
    if len(dose_values) < 2 or len(volume_values) == 0:
        return {}
    # 将bins转换为bin中心点，使其与volume_values长度一致
    # bin_centers = [(dose_values[i] + dose_values[i+1]) / 2 for i in range(len(dose_values)-1)]
    # 或者，如果bins已经是中心点，则直接使用
    # 通常 bins 是 N+1 个边界，counts 是 N 个值
    # 我们需要将 bins 转换为中心点，使其与 counts 对齐
    dose_centers = np.array([(dose_values[i] + dose_values[i+1]) / 2 for i in range(len(dose_values)-1)])
    volumes = np.array(volume_values)
    # 检查长度是否匹配
    if len(dose_centers) != len(volumes):
        print(f"  - 警告: 剂量和体积数组长度不匹配，无法计算DVH指标。")
        return {}

    metrics = {}
    structure_name_clean = structure_name.replace(' ', '_').replace('-', '_')

    if 'PTV' in structure_name_clean:
        metrics = find_DV_for_OAR(structure_name_clean, dose_centers, volumes)
        metrics.update(find_Dmax_for_OAR(structure_name_clean, dose_centers, volumes))
    elif structure_name_clean == 'Lungs':
        target_doses_gy = [5, 20] # Gy
        metrics = find_VD_for_OAR(structure_name_clean, target_doses_gy, dose_centers, volumes)
        metrics.update(find_Dmean_for_OAR(structure_name_clean, dose_centers, volumes))
    elif structure_name_clean == 'Heart':
        target_doses_gy = [4, 5] # Gy
        metrics = find_VD_for_OAR(structure_name_clean, target_doses_gy, dose_centers, volumes)
        metrics.update(find_Dmean_for_OAR(structure_name_clean, dose_centers, volumes))

    elif structure_name_clean == 'SpinalCord':
        metrics = find_Dmax_for_OAR(structure_name_clean, dose_centers, volumes)

    elif structure_name_clean == 'Esophagus':
        metrics = find_Dmax_for_OAR(structure_name_clean, dose_centers, volumes)
    elif structure_name_clean == 'Trachea':
        metrics = find_Dmax_for_OAR(structure_name_clean, dose_centers, volumes)
    elif structure_name_clean == 'Lung_R':
        target_doses_gy = [4, 17] # Gy
        metrics = find_VD_for_OAR(structure_name_clean, target_doses_gy, dose_centers, volumes)
        metrics.update(find_Dmean_for_OAR(structure_name_clean, dose_centers, volumes))
    elif structure_name_clean == 'Lung_L':
        target_doses_gy = [4] # Gy
        metrics = find_VD_for_OAR(structure_name_clean, target_doses_gy, dose_centers, volumes)
    elif structure_name_clean == 'Breast_L':
        target_doses_gy = [4] # Gy
        metrics = find_VD_for_OAR(structure_name_clean, target_doses_gy, dose_centers, volumes)
    elif structure_name_clean == 'Thyroid':
        metrics = find_Dmean_for_OAR(structure_name_clean, dose_centers, volumes)
    elif structure_name_clean == 'Humerus_head_R':
        metrics = find_Dmean_for_OAR(structure_name_clean, dose_centers, volumes)


    return metrics

def get_contour_data(structure_ds, roi_number):
    """从RT Structure Set中提取指定ROI的轮廓点"""
    contours = {}
    for sequence in structure_ds.ROIContourSequence:
        if sequence.ReferencedROINumber == roi_number:
            for contour in sequence.ContourSequence:
                z = contour.ContourData[2]
                if z not in contours:
                    contours[z] = []
                data = contour.ContourData
                points = np.array(data).reshape((len(data)//3, 3))
                contours[z].append(points)
    return contours

def point_in_polygon(polygon, point):
    """射线交叉法判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def create_voxel_mask_from_contours(contours, dose_ds):
    """为给定的轮廓创建一个体素化的3D掩码"""
    rows, cols = dose_ds.Rows, dose_ds.Columns
    pixel_spacing = dose_ds.PixelSpacing
    image_position_patient = dose_ds.ImagePositionPatient
    
    z_positions = []
    grid_frame_offset_vector = getattr(dose_ds, 'GridFrameOffsetVector', [0])
    for i in range(dose_ds.NumberOfFrames):
        z_pos = image_position_patient[2] + (grid_frame_offset_vector[i] if i < len(grid_frame_offset_vector) else i * abs(grid_frame_offset_vector[1] - grid_frame_offset_vector[0]))
        z_positions.append(z_pos)
    unique_zs = sorted(list(set(z_positions)))

    mask_3d = np.zeros((len(unique_zs), rows, cols), dtype=bool)

    for i, z in enumerate(unique_zs):
        if z in contours:
            x_coords = image_position_patient[0] + np.arange(rows) * pixel_spacing[0]
            y_coords = image_position_patient[1] + np.arange(cols) * pixel_spacing[1]
            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            
            slice_mask = np.zeros((rows, cols), dtype=bool)
            
            for contour_set in contours[z]:
                points_2d = contour_set[:, :2]
                
                for r in range(rows):
                    for c in range(cols):
                        if point_in_polygon(points_2d, (xx[r, c], yy[r, c])):
                            slice_mask[r, c] = True
                            break
            mask_3d[i, :, :] = slice_mask
            
    return mask_3d, unique_zs


def calculate_CI_GI(ds_struc, ds_plan, ds_dose, target_ROI, prescription_Dose_Gy):
    metrics = {}

    # 1. 找到目标ROI的编号
    target_roi_number = None
    for roi_structure in ds_struc.StructureSetROISequence:
        if roi_structure.ROIName.lower().strip() == target_ROI.lower().strip():
            target_roi_number = roi_structure.ROINumber
            break
    
    if target_roi_number is None:
        print(f"在结构文件中未找到名为 '{target_ROI}' 的ROI。")
        return {}

    # 2. 获取目标ROI的轮廓数据并创建掩码
    print("正在处理轮廓和创建掩码...")
    target_contours = get_contour_data(ds_struc, target_roi_number)
    target_mask_3d, unique_zs = create_voxel_mask_from_contours(target_contours, ds_dose)

    # 3. 获取剂量矩阵
    print("正在处理剂量矩阵...")
    dose_scaling = ds_dose.DoseGridScaling
    dose_matrix_gy = ds_dose.pixel_array.astype(np.float64) * dose_scaling

    # 4. 计算体素体积
    dx = abs(ds_dose.PixelSpacing[0]) / 10 # mm -> cm
    dy = abs(ds_dose.PixelSpacing[1]) / 10
    dz = abs(unique_zs[1] - unique_zs[0]) / 10 if len(unique_zs) > 1 else 1
    voxel_volume_cc = dx * dy * dz

    # 5. 计算所需体积
    print("正在计算体积...")
    # 靶区总体积
    target_vol_cc = np.sum(target_mask_3d) * voxel_volume_cc

    # CI相关体积
    #rx_mask_on_target = (dose_matri x_gy >= prescription_Dose_Gy) & target_mask_3d
    #target_vol_with_rx_cc = np.sum(rx_mask_on_target) * voxel_volume_cc

    rx_mask_all = dose_matrix_gy >= prescription_Dose_Gy
    total_vol_with_rx_cc = np.sum(rx_mask_all) * voxel_volume_cc

    # GI相关体积
    half_prescription_dose = prescription_Dose_Gy * 0.5
    rx_mask_half_prescription = dose_matrix_gy >= half_prescription_dose
    vol_half_prescription_cc = np.sum(rx_mask_half_prescription) * voxel_volume_cc

    # 6. 计算CI和GI
    #ci_conventional = target_vol_with_rx_cc / target_vol_cc if target_vol_cc > 0 else 0
    ci_rtog = total_vol_with_rx_cc / target_vol_cc if target_vol_cc > 0 else 0
    gi = vol_half_prescription_cc / total_vol_with_rx_cc if total_vol_with_rx_cc > 0 else 0

    metrics = {
        #"CI_Conventional": ci_conventional, #"CI_Conventional = V_target_with_Rx / V_target",
        "CI_RTOG": ci_rtog, #CI_RTOG = V_rx_isodose / V_target
        "Gradient_Index (GI)": gi, #"GI = V_half_prescription / V_prescription",
    }

    return metrics

def calc_plan_complexity(ds_struc, ds_plan, ds_dose, target_ROI,  prescription_Dose_Gy):
    # 计算MCS、GI、等评价计划复杂度的参数
    # PTV，
    results = calculate_CI_GI(ds_struc, ds_plan, ds_dose, target_ROI, prescription_Dose_Gy)
    return results


def process_patient_folder(folder_path, hospital):
    """
    处理单个病人文件夹中的DICOM文件。
    """
    print(f"正在处理文件夹: {folder_path}")
    rtplan_file = None
    rtdose_file = None
    rtstruct_file = None

    # 遍历文件夹，按模态分类文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.dcm') or filename.lower().endswith('.cdms'):
            filepath = os.path.join(folder_path, filename)
            try:
                temp_ds = dcmread(filepath, stop_before_pixels=True, force=True) # 先不加载像素数据以提高速度
                modality = getattr(temp_ds, 'Modality', None)
                if modality == 'RTPLAN':
                    rtplan_file = filepath
                elif modality == 'RTDOSE':
                    rtdose_file = filepath
                elif modality == 'RTSTRUCT':
                    rtstruct_file = filepath
            except Exception as e:
                print(f"  - 读取文件 {filename} 时出错: {e}")

    if not all([rtplan_file, rtdose_file, rtstruct_file]):
        print(f"  - 错误: 文件夹中缺少必要的DICOM文件 (RTPLAN, RTDOSE, RTSTRUCT)。跳过。")
        return None

    # 读取完整的DICOM数据集
    try:
        ds_plan = dcmread(rtplan_file, force=True)
        ds_dose = dcmread(rtdose_file, force=True)
        ds_struct = dcmread(rtstruct_file, force=True)
    except Exception as e:
        print(f"  - 读取DICOM文件时出错: {e}")
        return None

    patient_id = getattr(ds_plan.PatientID, 'value', 'N/A')
    patient_name = str(getattr(ds_plan.PatientName, 'family_name', 'N/A')) + ", " + str(getattr(ds_plan.PatientName, 'given_name', 'N/A'))
    #plan_label = getattr(ds_plan.RTPlanLabel, 'value', 'N/A')
    plan_label = os.path.basename(folder_path)
    
    # --- 提取计划基本信息 ---
    manufacturer = ds_plan.Manufacturer if hasattr(ds_plan, 'Manufacturer') else None
    Machinetype = ds_plan.BeamSequence[0].TreatmentMachineName
    TPStype = ds_plan.ManufacturerModelName if hasattr(ds_plan, 'ManufacturerModelName') else None
    num_beams = len(ds_plan.BeamSequence) if hasattr(ds_plan, 'BeamSequence') else 0
    
    total_mus = 0
    total_planned_dose_gy = 0

    if hasattr(ds_plan, 'BeamSequence'):
        
        # 更可靠的获取总MU的方法是从 Fraction Group
        if hasattr(ds_plan, 'FractionGroupSequence'):
            for fg in ds_plan.FractionGroupSequence:
                for rb in fg.ReferencedBeamSequence:
                #BeamMeterset是该射束的MU
                    if hasattr(rb, 'BeamMeterset'):#
                        total_mus += float(rb.BeamMeterset)
                    if hasattr(rb, 'BeamDose'):
                        total_planned_dose_gy += float(rb.BeamDose)
                break# 通常只处理第一个 FractionGroup


    total_fractions = 0
    if hasattr(ds_plan, 'FractionGroupSequence'):
        fg = ds_plan.FractionGroupSequence[0]
        total_fractions = fg.NumberOfFractionsPlanned

    if hasattr(ds_plan, 'DoseReferenceSequence'):
        planned_dose_gy = ds_plan.DoseReferenceSequence[0].TargetPrescriptionDose # 单位是 Gy
    else:
        planned_dose_gy = total_planned_dose_gy * total_fractions


    


    # --- 使用 dicompyler-core 解析文件并计算DVH ---
    target_ROI = 'PTV'
    OAR_for_Lung = ['PTV', 'Lungs', 'Heart', 'SpinalCord', 'Esophagus', 'Trachea']
    OAR_for_Breast = ['PTV_CW_EVALUATE', 'PTV_ALN', 'PTV_SCN', 'PTV_IMN', 'Humerus_Head_R', 'Breast_L', 'Thyroid', 'Lung_R', 'Lung_L', 'Heart']
    try:
        parser = dicomparser.DicomParser(ds_struct)
        structures = parser.GetStructures()
        # 创建一个包含三个文件路径的字典，传递给DVH计算器
        rtset_dict = {'rtss': ds_struct, 'rtdose': ds_dose, 'rtplan': ds_plan}
        
        # 这里我们遍历结构，为每个感兴趣的结构计算DVH
        calculated_dvhs = {}
        for roi_num, structure_data in structures.items():
            if roi_num == 0: continue # ROI 0 通常是无效的
            structure_name = structure_data['name']
            
            if structure_name in OAR_for_Lung or structure_name in OAR_for_Breast: #只计算特定的结构
                #print(f"  - 正在计算结构 '{structure_name}' 的DVH...")
                try:
                    # 使用 dicompyler-core 计算DVH
                    dvh = dvhcalc.get_dvh(ds_struct, ds_dose, roi_num)
                    calculated_dvhs[roi_num] = {'name': structure_name, 'dvh': dvh}
                except Exception as e:
                    print(f"    - 计算结构 '{structure_name}' 时出错: {e}")
                
    except Exception as e:
        print(f"  - 解析DICOM结构集或计算DVH时出错: {e}")
        return None

    # --- 组织输出数据 ---
    row_data = {
        'Hospital': hospital,
        'RTPlan': os.path.basename(rtplan_file),
        'RTDose': os.path.basename(rtdose_file),
        'RTStru': os.path.basename(rtstruct_file),
        #'PatientID': patient_id,
        'Manufacturer': manufacturer,
        'PlanLabel': plan_label,
        'Machine': Machinetype,
        'TPS': TPStype,
        'NumBeams': num_beams,
        'TotalMUs': round(total_mus, 2),
        'TotalFractions': total_fractions,
        'TotalPlannedDose_Gy': round(planned_dose_gy, 2),
        'DoseGridSpacing_mm': round(ds_dose.GridFrameOffsetVector[1] - ds_dose.GridFrameOffsetVector[0], 3) if len(getattr(ds_dose, 'GridFrameOffsetVector', [])) > 1 else 'N/A',
    }

    # 将所有计算出的DVH指标合并到行数据中
    """ if plan_label == '患者2':
        CI_GI_metrics = calc_plan_complexity(ds_struct, ds_plan, ds_dose, target_ROI, total_planned_dose_gy)
        row_data.update(CI_GI_metrics) """

    for roi_data in calculated_dvhs.values():
            struct_name = roi_data['name']
            dvh_obj = roi_data['dvh']
            struct_metrics = calculate_dvh_metrics(dvh_obj, struct_name)
            row_data.update(struct_metrics)
            #print(struct_metrics)
        
    return row_data

def main(data_directory, output_csv_path):
    """
    主函数，遍历数据目录下的所有病人文件夹。
    """
    all_patients_data = []

    for hos_folder in os.listdir(data_directory):
        hos_directory = os.path.join(data_directory, hos_folder)
        if os.path.isdir(hos_directory):
            for patient_folder in os.listdir(hos_directory):
                patient_full_path = os.path.join(hos_directory, patient_folder)
                if os.path.isdir(patient_full_path):
                    patient_row = process_patient_folder(patient_full_path, hos_folder)
                    if patient_row:
                        all_patients_data.append(patient_row)

    if not all_patients_data:
        print("没有找到任何有效的病人数据。")
        return

    df = pd.DataFrame(all_patients_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n所有数据已成功提取并保存到: {output_csv_path}. 共处理了 {len(all_patients_data)} 位病人.")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请修改这些路径为你的实际路径
    DATA_DIR = "/Users/xiangxia/Documents/Study/Data/From_Diff_hospital/PlanData" # 包含多个病人子文件夹的根目录
    OUTPUT_CSV = "extracted_radiation_data.csv"

    main(DATA_DIR, OUTPUT_CSV)