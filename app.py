import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json # 导入json库

# --- 1. 页面配置与文本库 ---

st.set_page_config(page_title="Cognitive Risk Predictor", layout="wide", initial_sidebar_state="expanded")

# [UI优化] 创建一个集中的文本库，用于中英双语切换
TEXTS = {
    "page_title": {"中文": "认知障碍风险预测工具", "English": "Cognitive Impairment Risk Prediction Tool"},
    "main_title": {"中文": "🧠 认知障碍 (MCI/AD) 风险预测工具", "English": "🧠 Cognitive Impairment (MCI/AD) Risk Prediction Tool"},
    
    # Sidebar
    "settings_header": {"中文": "⚙️ 设置", "English": "Settings"},
    "language_label": {"中文": "语言 / Language", "English": "Language / 語言"},
    "model_loaded": {"中文": "已加载模型", "English": "Loaded Model"},
    "about_header": {"中文": "ℹ️ 关于此工具", "English": "About This Tool"},
    "about_text": {
        "中文": "本工具基于已发表的研究成果，使用经过校准的机器学习模型，根据您的个人信息、临床指标和生活方式来预测未来患认知障碍（MCI/AD）的风险。所有结果仅供参考，不能替代专业医疗诊断。",
        "English": "This tool utilizes a calibrated machine learning model based on published research to predict the risk of future cognitive impairment (MCI/AD) based on your personal information, clinical markers, and lifestyle. All results are for reference only and cannot replace a professional medical diagnosis."
    },

    # Input Sections
    "personal_info_header": {"中文": "👤 个人基本信息", "English": "Personal Information"},
    "age": {"中文": "年龄 (Age)", "English": "Age"},
    "gender": {"中文": "性别 (Gender)", "English": "Gender"},
    "edu": {"中文": "受教育年限 (Education)", "English": "Years of Education"},
    "bmi": {"中文": "身体质量指数 (BMI)", "English": "Body Mass Index (BMI)"},
    
    "biomarkers_header": {"中文": "🩸 核心生物标志物", "English": "Core Biomarkers"},
    "abo": {"中文": "血清标志物 ABO (标准化前)", "English": "Serum Marker ABO (pre-standardization)"},
    "apoe4": {"中文": "APOE4 携带者", "English": "APOE4 Carrier"},

    "lifestyle_header": {"中文": "❤️ 生活方式与病史", "English": "Lifestyle & Medical History"},
    "lifestyle_subheader": {"中文": "生活方式 (Lifestyle)", "English": "Lifestyle"},
    "smoke": {"中文": "当前是否吸烟", "English": "Currently Smoking"},
    "alcohol": {"中文": "当前是否饮酒", "English": "Currently Drinking Alcohol"},
    "history_subheader": {"中文": "病史与家族史", "English": "Medical & Family History"},
    "hypertension": {"中文": "高血压病史", "English": "History of Hypertension"},
    "diabetes": {"中文": "糖尿病史", "English": "History of Diabetes"},
    "hyperlipidemia": {"中文": "高血脂病史", "English": "History of Hyperlipidemia"},
    "dementia_history": {"中文": "痴呆家族史", "English": "Family History of Dementia"},
    "depression_history": {"中文": "抑郁症家族史", "English": "Family History of Depression"},

    # Options
    "option_yes": {"中文": "是", "English": "Yes"},
    "option_no": {"中文": "否", "English": "No"},
    "gender_female": {"中文": "女性", "English": "Female"},
    "gender_male": {"中文": "男性", "English": "Male"},

    # Prediction
    "button_predict": {"中文": "📈 点击进行风险预测", "English": "📈 Predict Risk"},
    "predict_success": {"中文": "✅ 预测完成！", "English": "✅ Prediction Complete!"},
    "predict_header": {"中文": "认知障碍（MCI/AD）风险概率", "English": "Cognitive Impairment (MCI/AD) Risk Probability"},
    
    # Advice
    "advice_header": {"中文": "📋 认知健康建议", "English": "Cognitive Health Advice"},
    "risk_label_vh": {"中文": "风险评估：非常高", "English": "Risk Assessment: Very High"},
    "advice_vh": {
        "中文": "**核心建议**: 我们强烈建议您**立即咨询**神经科、老年科或精神心理科的专业医生，进行一次全面的认知功能评估和神经系统检查。\n\n**具体行动**: 请不要拖延，尽快预约门诊，并与医生分享您的担忧以及本模型的预测结果作为参考。",
        "English": "**Core Advice**: We strongly recommend that you **immediately consult** a neurologist, geriatrician, or psychiatrist for a comprehensive cognitive function assessment and neurological examination.\n\n**Action**: Please do not delay. Schedule an appointment as soon as possible and share your concerns and this model's prediction results with your doctor for reference."
    },
    "risk_label_h": {"中文": "风险评估：较高", "English": "Risk Assessment: High"},
    "advice_h": {
        "中文": "**核心建议**: 建议您主动咨询专业医生，讨论您的认知健康状况，并考虑进行认知功能筛查（如MoCA、MMSE量表）。\n\n**生活方式**: 请积极管理心血管健康（控制血压、血糖、血脂），增加体育锻炼和社交活动，保持大脑活跃（如阅读、学习新技能）。",
        "English": "**Core Advice**: We recommend you proactively consult a professional doctor to discuss your cognitive health and consider undergoing a cognitive function screening (e.g., MoCA, MMSE).\n\n**Lifestyle**: Actively manage your cardiovascular health (control blood pressure, blood sugar, lipids), increase physical exercise and social activities, and keep your brain active (e.g., reading, learning new skills)."
    },
    "risk_label_m": {"中文": "风险评估：中等", "English": "Risk Assessment: Medium"},
    "advice_m": {
        "中文": "**核心建议**: 建议您提高对认知健康的关注，在年度体检中加入认知功能相关的检查。\n\n**生活方式**: 保持均衡饮食（推荐地中海饮食），保证充足睡眠，进行规律的体育锻炼，并积极参与社交和脑力活动。",
        "English": "**Core Advice**: We recommend you increase your attention to cognitive health and consider adding cognitive-related checks to your annual physical examination.\n\n**Lifestyle**: Maintain a balanced diet (Mediterranean diet is recommended), ensure adequate sleep, engage in regular physical exercise, and actively participate in social and mental activities."
    },
    "risk_label_l": {"中文": "风险评估：较低", "English": "Risk Assessment: Low"},
    "advice_l": {
        "中文": "**核心建议**: 您的当前风险较低，这是一个非常好的信号。\n\n**生活方式**: 请继续保持您健康的生活习惯，包括均衡饮食、规律运动、充足睡眠和积极的社交生活，以长久维护您的大脑健康。",
        "English": "**Core Advice**: Your current risk is low, which is a very positive sign.\n\n**Lifestyle**: Please continue to maintain your healthy habits, including a balanced diet, regular exercise, adequate sleep, and an active social life, to preserve your brain health long-term."
    },

    # Disclaimer
    "disclaimer": {
        "中文": "**免责声明**: 本工具的预测结果仅供参考，不能替代专业的医疗诊断。所有健康相关的决策，请务必咨询您的医生。",
        "English": "**Disclaimer**: The prediction results of this tool are for reference only and cannot replace professional medical diagnosis. For all health-related decisions, please be sure to consult your doctor."
    }
}

# --- 会话状态初始化 ---
if 'lang' not in st.session_state:
    st.session_state.lang = "中文"

# --- 2. 加载模型 ---
MODEL_DIR = Path("./machine_learning_results_MCI_AD")
@st.cache_resource
def load_model():
    """动态加载最佳模型及所有预处理器"""
    try:
        # [核心修正] 动态加载最佳模型
        # 1. 读取记录最佳模型名称的文件
        with open(MODEL_DIR / 'best_model_info.json', 'r') as f:
            best_model_info = json.load(f)
        best_model_name = best_model_info['best_model_name']
        
        # 2. 根据名称构建模型文件名并加载
        model_filename = f'final_calibrated_{best_model_name.lower()}_model.joblib'
        model = joblib.load(MODEL_DIR / model_filename)
        
        # 加载其他文件
        imputer = joblib.load(MODEL_DIR / 'imputer.joblib')
        scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
        model_columns = joblib.load(MODEL_DIR / 'model_columns.joblib')
        continuous_cols = ['edu', 'ABO', 'age', 'BMI']
        imputer_columns = ['edu', 'ABO', 'dia', 'APOE4_carrier', 'age', 'gender', 'BMI', 'smoke', 'alcohol', 'dementia_family_history', 'depression_family_history', 'hypertension', 'diabetes', 'hyperlipidemia']
        
        # 在侧边栏显示加载的模型名称，方便调试和确认
        st.sidebar.info(f"{TEXTS['model_loaded'][st.session_state.lang]}: **{best_model_name}**")
        
        # 动态更新“关于”文本中的模型名称
        TEXTS["about_text"]["中文"] = TEXTS["about_text"]["中文"].replace("XGBoost", best_model_name)
        TEXTS["about_text"]["English"] = TEXTS["about_text"]["English"].replace("XGBoost", best_model_name)

        return model, imputer, scaler, model_columns, continuous_cols, imputer_columns
        
    except FileNotFoundError as e:
        st.error(f"Error: Loading model files failed. Please ensure all required .joblib and .json files are in the '{MODEL_DIR}' folder.")
        st.error(f"Specific error: {e}")
        return None, None, None, None, None, None

model, imputer, scaler, model_columns, continuous_cols, imputer_columns = load_model()

# --- 3. 侧边栏 ---
with st.sidebar:
    st.title(TEXTS["settings_header"][st.session_state.lang])
    
    selected_lang = st.radio(
        label=TEXTS["language_label"][st.session_state.lang],
        options=["中文", "English"],
        index=["中文", "English"].index(st.session_state.lang),
        horizontal=True
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()

    with st.expander(TEXTS["about_header"][st.session_state.lang]):
        st.write(TEXTS["about_text"][st.session_state.lang])

# --- 4. 主页面 ---
st.title(TEXTS["main_title"][st.session_state.lang])
st.markdown("---")

if model:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.header(TEXTS["personal_info_header"][st.session_state.lang])
            age = st.number_input(TEXTS["age"][st.session_state.lang], 18, 120, 70, 1)
            gender = st.radio(TEXTS["gender"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["gender_female"][st.session_state.lang] if x == 0 else TEXTS["gender_male"][st.session_state.lang], horizontal=True)
            edu = st.number_input(TEXTS["edu"][st.session_state.lang], 0, 40, 12, 1)
            bmi = st.number_input(TEXTS["bmi"][st.session_state.lang], 15.0, 50.0, 24.0, 0.1, format="%.1f")

    with col2:
        with st.container(border=True):
            st.header(TEXTS["biomarkers_header"][st.session_state.lang])
            abo = st.number_input(TEXTS["abo"][st.session_state.lang], 0.0, 500.0, 100.0, 0.1, format="%.1f")
            apoe4_carrier = st.radio(TEXTS["apoe4"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            
    with col3:
        with st.container(border=True):
            st.header(TEXTS["lifestyle_header"][st.session_state.lang])
            st.subheader(TEXTS["lifestyle_subheader"][st.session_state.lang])
            smoke = st.radio(TEXTS["smoke"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            alcohol = st.radio(TEXTS["alcohol"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            st.subheader(TEXTS["history_subheader"][st.session_state.lang])
            hypertension = st.radio(TEXTS["hypertension"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            diabetes = st.radio(TEXTS["diabetes"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            hyperlipidemia = st.radio(TEXTS["hyperlipidemia"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            dementia_family_history = st.radio(TEXTS["dementia_history"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
            depression_family_history = st.radio(TEXTS["depression_history"][st.session_state.lang], [0, 1], format_func=lambda x: TEXTS["option_no"][st.session_state.lang] if x == 0 else TEXTS["option_yes"][st.session_state.lang], horizontal=True)
    
    st.markdown("---")

    # --- 5. 预测逻辑 ---
    if st.button(TEXTS["button_predict"][st.session_state.lang], use_container_width=True, type="primary"):
        input_data = {'edu': edu, 'ABO': abo, 'APOE4_carrier': apoe4_carrier, 'age': age, 'gender': gender, 'BMI': bmi, 'smoke': smoke, 'alcohol': alcohol, 'dementia_family_history': dementia_family_history, 'depression_family_history': depression_family_history, 'hypertension': hypertension, 'diabetes': diabetes, 'hyperlipidemia': hyperlipidemia}
        input_df_features = pd.DataFrame([input_data])
        
        # 预处理流程 (与之前版本相同)
        input_for_imputer = pd.DataFrame(columns=imputer_columns)._append(input_df_features, ignore_index=True)
        input_imputed_values = imputer.transform(input_for_imputer)
        input_imputed_df = pd.DataFrame(input_imputed_values, columns=imputer_columns)
        input_features_processed = input_imputed_df.drop('dia', axis=1)
        input_features_processed = input_features_processed[model_columns]
        input_scaled_df = input_features_processed.copy()
        input_scaled_df[continuous_cols] = scaler.transform(input_features_processed[continuous_cols])
        
        # 进行预测
        prediction_proba = model.predict_proba(input_scaled_df)[:, 1]
        risk_percentage = prediction_proba[0] * 100
        
        st.success(f"**{TEXTS['predict_success'][st.session_state.lang]}**")
        st.metric(label=TEXTS['predict_header'][st.session_state.lang], value=f"{risk_percentage:.2f} %")
        st.progress(int(risk_percentage))

        with st.expander(TEXTS["advice_header"][st.session_state.lang], expanded=True):
            if risk_percentage > 75:
                st.error(f"**{TEXTS['risk_label_vh'][st.session_state.lang]}**")
                st.write(TEXTS["advice_vh"][st.session_state.lang])
            elif risk_percentage > 50:
                st.warning(f"**{TEXTS['risk_label_h'][st.session_state.lang]}**")
                st.write(TEXTS["advice_h"][st.session_state.lang])
            elif risk_percentage > 25:
                st.info(f"**{TEXTS['risk_label_m'][st.session_state.lang]}**")
                st.write(TEXTS["advice_m"][st.session_state.lang])
            else:
                st.success(f"**{TEXTS['risk_label_l'][st.session_state.lang]}**")
                st.write(TEXTS["advice_l"][st.session_state.lang])

    st.caption(TEXTS["disclaimer"][st.session_state.lang])
