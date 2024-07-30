import pickle
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler


#load model
osteoporosis_model = pickle.load(open('osteoporosis_model.sav','rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
#with open('osteoporosis_model.sav', 'rb') as model_file:

#judul web
st.title('Prediksi Resiko Osteoporosis')
# Membuat dua kolom untuk input
col1, col2, col3 = st.columns(3)

# Input data dari pengguna pada kolom pertama
with col1:
    age = st.number_input('Umur', min_value=0, max_value=120, step=1)
    gender = st.selectbox('Jenis Kelamin', ('Laki-laki', 'Perempuan'))
    family_history = st.selectbox('Riwayat Keluarga', ('Ya', 'Tidak'))
    race_ethnicity = st.selectbox('Ras/Etnis', ('Afrika-Amerika', 'Asia', 'Kaukasia'))
    medications = st.selectbox('Pengobatan', ('Penggunaan steroid', 'Tidak'))
   

# Input data dari pengguna pada kolom kedua
with col2:
    prior_fractures = st.selectbox('Riwayat Patah Tulang Sebelumnya', ('Ya', 'Tidak'))
    hormonal_changes = st.selectbox('Perubahan Hormonal', ('Normal', 'Postmenopausal'))
    body_weight = st.selectbox('Berat Badan', ('Normal', 'Underweight'))
    calcium_intake = st.selectbox('Asupan Kalsium', ('Rendah', 'Sedang'))
    
    
    
# Input data dari pengguna pada kolom ketiga
with col3:
    vitamin_d_intake = st.selectbox('Asupan Vitamin D', ('Rendah', 'Sedang'))
    physical_activity = st.selectbox('Aktivitas Fisik', ('Aktif', 'Kurang Aktif'))
    smoking = st.selectbox('Merokok', ('Ya', 'Tidak'))
    alcohol_consumption = st.selectbox('Konsumsi Alkohol', ('Ya', 'Tidak'))
    medical_conditions = st.selectbox('Kondisi Medis', ('Hipertiroidisme', 'Tidak Ada', 'Artritis Reumatoid'))

# Konversi input ke dalam bentuk numerik
gender_numeric = 0 if gender == 'Perempuan' else 1
hormonal_changes_numeric = 1 if hormonal_changes == 'Postmenopausal' else 0
family_history_numeric = 1 if family_history == 'Ya' else 0

#Konversi ras/etnis ke dalam kolom biner
race_ethnicity_african_american = 1 if race_ethnicity == 'Afrika-Amerika' else 0
race_ethnicity_asian = 1 if race_ethnicity == 'Asia' else 0
race_ethnicity_caucasian = 1 if race_ethnicity == 'Kaukasia' else 0

#Konversi kondisi medis ke dalam kolom biner
medical_conditions_hyperthyroidism = 1 if medical_conditions == 'Hipertiroidisme' else 0
medical_conditions_no = 1 if medical_conditions == 'Tidak Ada' else 0
medical_conditions_rheumatoid_arthritis = 1 if medical_conditions == 'Artritis Reumatoid' else 0


body_weight_numeric = 1 if body_weight == 'Underweight' else 0
calcium_intake_numeric = 1 if calcium_intake == 'Rendah' else 0
vitamin_d_intake_numeric = 1 if vitamin_d_intake == 'Sedang' else 0
physical_activity_numeric = 1 if physical_activity == 'Kurang Aktif' else 0
smoking_numeric = 1 if smoking == 'Ya' else 0
alcohol_consumption_numeric = 1 if alcohol_consumption == 'Ya' else 0
medications_numeric = 0 if medications == 'Tidak' else 1
prior_fractures_numeric = 1 if prior_fractures == 'Ya' else 0

# Siapkan data input untuk model
input_data = np.array([[age, gender_numeric, hormonal_changes_numeric, family_history_numeric, 
                        body_weight_numeric, calcium_intake_numeric, vitamin_d_intake_numeric, physical_activity_numeric,
                        smoking_numeric, alcohol_consumption_numeric, medications_numeric, prior_fractures_numeric, 
                        race_ethnicity_african_american, race_ethnicity_asian, race_ethnicity_caucasian, 
                        medical_conditions_hyperthyroidism, medical_conditions_no, medical_conditions_rheumatoid_arthritis]])

# Normalisasi data input
input_data_scaled = scaler.transform(input_data)

# Tambahkan tombol untuk membuat prediksi
if st.button('Prediksi'):
    # Buat prediksi
    prediction = osteoporosis_model.predict(input_data_scaled)
    
    # Tampilkan hasilnya
    st.write('Prediksi:', 'Anda memiliki risiko osteoporosis' if prediction == 1 else 'Anda tidak memiliki risiko osteoporosis')