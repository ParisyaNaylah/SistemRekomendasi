import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sistem Rekomendasi Meal Plan",
    page_icon="🥗",
    layout="wide"
)

@st.cache_data
def load_and_preprocess():
    df_usda = pd.read_excel("usda.xlsx")
    food_group_mapping = {
        'Daging dan Sosis'             : 'Makanan Utama',
        'Daging Merah dan Hasil Buruan': 'Makanan Utama',
        'Ikan dan Kerang'              : 'Makanan Utama',
        'Sereal dan Pasta'             : 'Makanan Utama',
        'Unggas'                       : 'Makanan Utama',
        'Keju, Susu, dan Telur'        : 'Makanan Pendamping',
        'Makanan Pembuka dan Lauk'     : 'Makanan Pendamping',
        'Produk Panggang'              : 'Makanan Pendamping',
        'Sayuran'                      : 'Makanan Pendamping',
        'Sup dan Kaldu'                : 'Makanan Pendamping',
        'Buah dan Jus'                 : 'Minuman dan Buah',
        'Minuman'                      : 'Minuman dan Buah',
    }
    df_clean = df_usda[~df_usda['FoodName'].str.contains(
        'mentah|tidak siap|alkohol|babi|leavening|baking soda|baking powder|'
        'ragi|bir|keras|pengganti krim', case=False, na=False)].copy()
    df_clean = df_clean[df_clean['FoodCategory'] != 'Babi']
    df_clean['FoodGroup'] = df_clean['FoodCategory'].map(food_group_mapping)
    cols = df_clean.columns.tolist()
    food_name_idx = cols.index('FoodName')
    if 'FoodGroup' in cols:
        cols.remove('FoodGroup')
    cols.insert(food_name_idx, 'FoodGroup')
    df_clean = df_clean[cols]
    df_clean = df_clean.fillna(0)
    df_clean = df_clean.drop_duplicates(subset=['FoodName'])
    selected_columns = [
        'FoodID', 'FoodCategory', 'FoodGroup', 'FoodName',
        'Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Serat',
        'GulaTotal', 'LemakJenuh',
        'Natrium', 'Kalium', 'Kalsium', 'Magnesium',
        'Besi', 'VitaminC', 'VitaminD'
    ]
    df_clean = df_clean[selected_columns]
    df_original = df_clean.copy()
    non_numeric_cols = ['FoodID', 'FoodCategory', 'FoodGroup', 'FoodName']
    numeric_cols = [col for col in df_clean.columns if col not in non_numeric_cols]
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    return df_clean, scaler, df_original, numeric_cols

df_preprocessed, scaler, df_original, nutrient_cols = load_and_preprocess()

def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2

def get_bmi_category(bmi):
    if bmi < 18.5:   return "Berat Badan Kurang (Underweight)"
    elif bmi < 25:   return "Berat Badan Normal"
    elif bmi < 30:   return "Berat Badan Berlebih (Overweight)"
    elif bmi < 35:   return "Obesitas I"
    elif bmi < 40:   return "Obesitas II"
    else:            return "Obesitas III"

def calculate_energy_needs(gender, age, height, weight, activity_level):
    if gender.lower() == 'laki-laki':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    factors = {
        "sedentary": 1.2, "lightly active": 1.375,
        "moderately active": 1.55, "very active": 1.725, "extra active": 1.9
    }
    return bmr * factors[activity_level]

def get_age_category(age):
    if 10 <= age <= 18: return "remaja"
    elif 19 <= age <= 64: return "dewasa"
    else: return "lansia"

def calculate_weight_loss_plan(current_weight, target_weight, duration_months, energy_needs):
    total_loss      = current_weight - target_weight
    total_deficit   = total_loss * 7700
    duration_days   = duration_months * 30
    daily_deficit   = total_deficit / duration_days
    adjusted_energy = energy_needs - daily_deficit
    return {
        'total_loss_target'   : total_loss,
        'total_deficit_needed': total_deficit,
        'daily_deficit'       : daily_deficit,
        'adjusted_energy'     : adjusted_energy,
    }

def validate_weight_loss_plan(gender, current_weight, target_weight,
                               duration_months, energy_needs, adjusted_energy):
    MIN_ENERGY       = 1500 if gender.lower() == 'laki-laki' else 1200
    MAX_MONTHLY_LOSS = 4
    warnings_list    = []
    suggestion       = None
    if target_weight >= current_weight:
        warnings_list.append("Target berat badan harus lebih rendah.")
        return False, warnings_list, suggestion
    total_loss   = current_weight - target_weight
    monthly_loss = total_loss / duration_months
    if monthly_loss > MAX_MONTHLY_LOSS or adjusted_energy < MIN_ENERGY:
        warnings_list.append("Target terlalu ekstrem.")
        max_safe = energy_needs - MIN_ENERGY
        if max_safe > 0:
            min_months = max((total_loss * 7700 / max_safe) / 30, total_loss / MAX_MONTHLY_LOSS)
            safe       = math.ceil(min_months * 2) / 2
            suggestion = f"Untuk menurunkan {total_loss} kg dengan aman, disarankan durasi minimal {safe} bulan."
        else:
            suggestion = "Defisit kalori tidak disarankan karena energi sudah di batas minimum."
    return len(warnings_list) == 0, warnings_list, suggestion

def get_nutrition_requirements(gender, age, adjusted_energy):
    age_category = get_age_category(age)
    idx = 0
    if gender.lower() == 'laki-laki' and age_category == 'remaja':
        if age <= 12: idx = 0
        elif age <= 15: idx = 1
        else: idx = 2
    elif gender.lower() == 'laki-laki' and age_category == 'dewasa':
        if age <= 29: idx = 3
        elif age <= 49: idx = 4
        else: idx = 5
    elif gender.lower() == 'laki-laki' and age_category == 'lansia':
        idx = 6 if age <= 80 else 7
    elif gender.lower() == 'perempuan' and age_category == 'remaja':
        if age <= 12: idx = 8
        elif age <= 15: idx = 9
        else: idx = 10
    elif gender.lower() == 'perempuan' and age_category == 'dewasa':
        if age <= 29: idx = 11
        elif age <= 49: idx = 12
        else: idx = 13
    elif gender.lower() == 'perempuan' and age_category == 'lansia':
        idx = 14 if age <= 80 else 15
    return {
        'Energi'     : adjusted_energy,
        'Protein'    : (25 * adjusted_energy / 100) / 4,
        'Lemak'      : (25 * adjusted_energy / 100) / 9,
        'Karbohidrat': (50 * adjusted_energy / 100) / 4,
        'Serat'      : [28,34,37,37,36,30,25,22,27,29,29,32,30,25,22,20][idx],
        'Natrium'    : [1300,1500,1700,1500,1500,1300,1100,1000,1400,1500,1600,1500,1500,1400,1200,1000][idx],
        'Kalium'     : [3900,4800,5300,4700,4700,4700,4700,4700,4400,4800,5000,4700,4700,4700,4700,4700][idx],
        'Kalsium'    : [1200,1200,1200,1000,1000,1200,1200,1200,1200,1200,1200,1000,1000,1200,1200,1200][idx],
        'Magnesium'  : [160,225,270,360,360,360,350,350,170,220,230,330,340,340,320,320][idx],
        'Besi'       : [8,11,11,9,9,9,9,9,8,15,15,18,18,8,8,8][idx],
        'VitaminC'   : [50,75,90,90,90,90,90,90,50,65,75,75,75,75,75,75][idx],
        'VitaminD'   : [15,15,15,15,15,15,20,20,15,15,15,15,15,15,20,20][idx],
    }

def process_user_data(gender, age, height, weight, activity_level, target_weight, duration_months):
    bmi          = calculate_bmi(weight, height)
    bmi_category = get_bmi_category(bmi)
    if bmi < 25:
        return {
            'is_valid' : False,
            'warnings' : [f"Sistem ini tidak direkomendasikan untuk pengguna dengan kategori {bmi_category}."],
            'suggestion': None,
        }
    energy_needs     = calculate_energy_needs(gender, age, height, weight, activity_level)
    wlp              = calculate_weight_loss_plan(weight, target_weight, duration_months, energy_needs)
    is_valid, warnings_list, suggestion = validate_weight_loss_plan(
        gender, weight, target_weight, duration_months, energy_needs, wlp['adjusted_energy'])
    if not is_valid:
        return {'is_valid': False, 'warnings': warnings_list, 'suggestion': suggestion}
    nutrition_req = get_nutrition_requirements(gender, age, wlp['adjusted_energy'])
    age_category  = get_age_category(age)
    return {
        'is_valid'         : True,
        'warnings'         : warnings_list,
        'gender'           : gender,
        'age'              : age,
        'age_category'     : age_category,
        'age_with_category': f"{age} tahun ({age_category})",
        'height'           : height,
        'weight'           : weight,
        'bmi'              : round(bmi, 2),
        'bmi_category'     : bmi_category,
        'activity_level'   : activity_level,
        'target_weight'    : target_weight,
        'duration_months'  : duration_months,
        'energy_needs'     : round(energy_needs, 2),
        'weight_loss_plan' : {
            'total_loss_target'   : wlp['total_loss_target'],
            'total_deficit_needed': round(wlp['total_deficit_needed'], 2),
            'daily_deficit'       : round(wlp['daily_deficit'], 2),
            'adjusted_energy'     : round(wlp['adjusted_energy'], 2),
        },
        'nutrition_req': nutrition_req,
    }

KNN_FEATURES = [
    'Protein', 'Lemak', 'Karbohidrat', 'Serat',
    'Natrium', 'Kalium', 'Kalsium', 'Magnesium',
    'Besi', 'VitaminC', 'VitaminD'
]
MEAL_CALORIE_RATIO = {
    'Sarapan'    : 0.25,
    'Snack Pagi' : 0.10,
    'Makan Siang': 0.35,
    'Snack Sore' : 0.10,
    'Makan Malam': 0.20,
}
MEAL_FOOD_GROUPS = {
    'Sarapan'    : ['Makanan Utama', 'Makanan Pendamping'],
    'Snack Pagi' : ['Minuman dan Buah', 'Kacang'],
    'Makan Siang': ['Makanan Utama', 'Makanan Pendamping', 'Minuman dan Buah'],
    'Snack Sore' : ['Minuman dan Buah', 'Kacang'],
    'Makan Malam': ['Makanan Utama', 'Makanan Pendamping'],
}
N_CANDIDATES = 100

def build_category_rotation(df_original, days=7, seed=42):
    rng = random.Random(seed)
    categories_per_group = {}
    for fg in df_original['FoodGroup'].dropna().unique():
        cats = df_original[df_original['FoodGroup']==fg]['FoodCategory'].dropna().unique().tolist()
        categories_per_group[fg] = cats
    rotation_order = {}
    for fg, cats in categories_per_group.items():
        shuffled = cats.copy(); rng.shuffle(shuffled)
        while len(shuffled) < days:
            extra = cats.copy(); rng.shuffle(extra); shuffled.extend(extra)
        rotation_order[fg] = shuffled
    rotation = {}
    for day in range(1, days+1):
        rotation[day] = {fg: rotation_order[fg][day-1] for fg in categories_per_group}
    return rotation

def get_knn_candidates_per_group(nutrition_req, df_preprocessed, df_original,
                                  scaler, nutrient_cols, food_group,
                                  meal_proportion, n_candidates=N_CANDIDATES,
                                  target_category=None):
    if food_group == 'Kacang':
        mask          = df_preprocessed['FoodName'].str.startswith('Kacang', na=False)
        df_group      = df_preprocessed[mask]
        df_orig_group = df_original[mask]
    else:
        if target_category:
            mask = ((df_preprocessed['FoodGroup']==food_group) &
                    (df_preprocessed['FoodCategory']==target_category))
        else:
            mask = df_preprocessed['FoodGroup']==food_group
        df_group      = df_preprocessed[mask]
        df_orig_group = df_original[mask]
    if df_group.empty:
        return pd.DataFrame()
    query_raw = [
        nutrition_req[col]*meal_proportion if col in nutrition_req else 0.0
        for col in nutrient_cols
    ]
    query_normalized = np.clip(scaler.transform([query_raw]), 0, 1)
    feat_indices     = [nutrient_cols.index(f) for f in KNN_FEATURES if f in nutrient_cols]
    query_knn        = query_normalized[0][feat_indices].reshape(1, -1)
    X_group  = df_group[KNN_FEATURES].values
    k_actual = min(n_candidates, len(df_group))
    knn_g    = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    knn_g.fit(X_group)
    distances, local_idx = knn_g.kneighbors(query_knn)
    candidates = df_orig_group.iloc[local_idx[0]].copy()
    candidates['euclidean_distance'] = distances[0]
    return candidates.sort_values('euclidean_distance', ascending=True)

def knapsack_select_food(candidates_df, calorie_budget):
    if candidates_df.empty:
        return None
    feasible = candidates_df[candidates_df['Energi'] <= calorie_budget].copy()
    if feasible.empty:
        return None
    feasible['knapsack_value'] = 1.0 / (feasible['euclidean_distance'] + 1e-6)
    max_val = feasible['knapsack_value'].max()
    max_cal = feasible['Energi'].max()
    max_fat = feasible['Lemak'].max()
    feasible['combined_score'] = (
        0.50 * (feasible['knapsack_value'] / (max_val + 1e-9)) +
        0.50 * (feasible['Energi']         / (max_cal + 1e-9)) -
        0.15 * (feasible['Lemak']          / (max_fat + 1e-9))
    )
    return feasible.loc[feasible['combined_score'].idxmax()].to_dict()

def generate_meal_plan(nutrition_req, df_preprocessed, df_original,
                       scaler, nutrient_cols, days=7):
    daily_calorie     = nutrition_req['Energi']
    used_foods        = set()
    meal_plan         = []
    category_rotation = build_category_rotation(df_original, days=days)
    for day in range(1, days+1):
        daily_plan    = {'Hari': day, 'Total_Kalori': 0, 'Makanan': []}
        total_day_cal = 0.0
        for meal_time, ratio in MEAL_CALORIE_RATIO.items():
            remaining      = daily_calorie * ratio
            allowed_groups = MEAL_FOOD_GROUPS[meal_time]
            for food_group in allowed_groups:
                if remaining <= 10:
                    continue
                target_category = None if food_group == 'Kacang' \
                                   else category_rotation[day].get(food_group)
                candidates = get_knn_candidates_per_group(
                    nutrition_req, df_preprocessed, df_original, scaler, nutrient_cols,
                    food_group, ratio, N_CANDIDATES, target_category)
                pool = candidates[~candidates['FoodName'].isin(used_foods)].copy() \
                       if not candidates.empty else pd.DataFrame()
                if pool.empty and food_group != 'Kacang':
                    candidates = get_knn_candidates_per_group(
                        nutrition_req, df_preprocessed, df_original, scaler, nutrient_cols,
                        food_group, ratio, N_CANDIDATES, None)
                    pool = candidates[~candidates['FoodName'].isin(used_foods)].copy() \
                           if not candidates.empty else pd.DataFrame()
                if pool.empty and food_group == 'Makanan Utama':
                    candidates = get_knn_candidates_per_group(
                        nutrition_req, df_preprocessed, df_original, scaler, nutrient_cols,
                        food_group, ratio, N_CANDIDATES, None)
                    pool = candidates.copy() if not candidates.empty else pd.DataFrame()
                if pool.empty:
                    continue
                chosen = knapsack_select_food(pool, remaining)
                if chosen is None:
                    continue
                used_foods.add(chosen['FoodName'])
                remaining     -= chosen['Energi']
                total_day_cal += chosen['Energi']
                daily_plan['Makanan'].append({
                    'WaktuMakan'       : meal_time,
                    'FoodGroup'        : chosen.get('FoodGroup', food_group),
                    'FoodID'           : chosen['FoodID'],
                    'FoodName'         : chosen['FoodName'],
                    'FoodCategory'     : chosen['FoodCategory'],
                    'Porsi'            : 100,
                    'Energi'           : round(chosen['Energi'], 2),
                    'Protein'          : round(chosen['Protein'], 2),
                    'Lemak'            : round(chosen['Lemak'], 2),
                    'Karbohidrat'      : round(chosen['Karbohidrat'], 2),
                    'Serat'            : round(chosen['Serat'], 2),
                    'Natrium'          : round(chosen['Natrium'], 2),
                    'Kalium'           : round(chosen['Kalium'], 2),
                    'Kalsium'          : round(chosen['Kalsium'], 2),
                    'Magnesium'        : round(chosen['Magnesium'], 2),
                    'Besi'             : round(chosen['Besi'], 2),
                    'VitaminC'         : round(chosen['VitaminC'], 2),
                    'VitaminD'         : round(chosen['VitaminD'], 2),
                    'EuclideanDistance': round(chosen['euclidean_distance'], 4),
                })
        min_cal    = daily_calorie * 0.90
        max_cal    = daily_calorie * 1.10
        target_cal = daily_calorie
        nutrisi_cols = ['Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Serat',
                        'GulaTotal', 'LemakJenuh', 'Natrium',
                        'Kalium', 'Kalsium', 'Magnesium', 'Besi', 'VitaminC', 'VitaminD']
        utama_pendamping = [f for f in daily_plan['Makanan']
                            if f['FoodGroup'] in ('Makanan Utama', 'Makanan Pendamping')
                            and f['WaktuMakan'] not in ('Snack Pagi', 'Snack Sore')]
        utama_pendamping.sort(key=lambda f: (0 if f['FoodGroup']=='Makanan Utama' else 1))
        satu_porsi = {f['FoodName']: {n: f[n] for n in nutrisi_cols if n in f}
                      for f in utama_pendamping}
        MAX_PORSI   = 300
        slot_totals = {}
        for f in daily_plan['Makanan']:
            slot_totals[f['WaktuMakan']] = slot_totals.get(f['WaktuMakan'], 0) + f['Energi']
        while total_day_cal < min_cal:
            kandidat = []
            for f in utama_pendamping:
                if f.get('Porsi', 100) >= MAX_PORSI: continue
                porsi_cal = satu_porsi[f['FoodName']]['Energi']
                if total_day_cal + porsi_cal > max_cal: continue
                slot = f['WaktuMakan']
                if slot_totals.get(slot,0)+porsi_cal > target_cal*MEAL_CALORIE_RATIO.get(slot,0.2)*1.3:
                    continue
                kandidat.append(f)
            if not kandidat: break
            target_food = min(kandidat,
                key=lambda f: (0 if f['FoodGroup']=='Makanan Utama' else 1,
                               satu_porsi[f['FoodName']]['Energi']))
            porsi_awal = satu_porsi[target_food['FoodName']]
            total_day_cal += porsi_awal['Energi']
            slot_totals[target_food['WaktuMakan']] = \
                slot_totals.get(target_food['WaktuMakan'],0) + porsi_awal['Energi']
            target_food['Porsi'] = target_food.get('Porsi', 100) + 100
            for nut in nutrisi_cols:
                if nut in target_food and nut in porsi_awal:
                    target_food[nut] = round(target_food[nut] + porsi_awal[nut], 2)
        daily_plan['Total_Kalori'] = round(total_day_cal, 2)
        meal_plan.append(daily_plan)
    return meal_plan

def meal_plan_to_df(meal_plan):
    rows = []
    for day_plan in meal_plan:
        for food in day_plan['Makanan']:
            rows.append({'Hari': day_plan['Hari'], **food})
    return pd.DataFrame(rows)

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🥗 Sistem Rekomendasi Menu Makanan Penurunan Berat Badan")

with st.sidebar:
    st.header("📋 Data Pengguna")
    gender = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
    age    = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=30)
    height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=300, value=100)
    activity_options = {
        "Sedentary"        : "sedentary",
        "Lightly Active"   : "lightly active",
        "Moderately Active": "moderately active",
        "Very Active"      : "very active",
        "Extra Active"     : "extra active",
    }
    activity_label = st.selectbox("Tingkat Aktivitas", list(activity_options.keys()), index=2)
    activity_level = activity_options[activity_label]
    target_weight   = st.number_input("Target Berat Badan (kg)", min_value=30, max_value=300, value=75)
    duration_months = st.number_input("Durasi (bulan)", min_value=1.0, max_value=36.0, value=6.5, step=0.5)
    generate_btn = st.button("🚀 Generate Meal Plan", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner("Memproses data pengguna..."):
        user_input = process_user_data(gender, age, height, weight,
                                       activity_level, target_weight, duration_months)
    if not user_input['is_valid']:
        st.error("❌ " + " ".join(user_input['warnings']))
        if user_input['suggestion']:
            st.info(f"💡 {user_input['suggestion']}")
    else:
        nutrition_req = user_input['nutrition_req']
        wlp           = user_input['weight_loss_plan']
        nutrient_targets = {
            'Energi'     : nutrition_req['Energi'],
            'Protein'    : nutrition_req['Protein'],
            'Lemak'      : nutrition_req['Lemak'],
            'Karbohidrat': nutrition_req['Karbohidrat'],
            'Serat'      : nutrition_req['Serat'],
            'Natrium'    : nutrition_req['Natrium'],
        }
        nutrient_cols_eval = list(nutrient_targets.keys())

        # --- PROFIL PENGGUNA ---
        st.header("👤 Profil Pengguna")
        prediksi_turun = (wlp['daily_deficit'] * duration_months * 30) * 0.00013
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("BMI", f"{user_input['bmi']:.1f}", user_input['bmi_category'])
        c2.metric("Kebutuhan Energi", f"{user_input['energy_needs']:.0f} kkal")
        c3.metric("Target Kalori Harian", f"{wlp['adjusted_energy']:.0f} kkal")
        c4.metric("Defisit", f"{wlp['daily_deficit']:.0f} kkal/hari")
        c5.metric("Penurunan", f"{prediksi_turun:.2f} kg",
                  f"dalam {duration_months} bulan")
        with st.expander("📊 Detail Kebutuhan Nutrisi Harian"):
            nutrisi_display = [
                ("Energi",      nutrition_req['Energi'],      "kkal"),
                ("Protein",     nutrition_req['Protein'],     "g"),
                ("Lemak",       nutrition_req['Lemak'],       "g"),
                ("Karbohidrat", nutrition_req['Karbohidrat'], "g"),
                ("Serat",       nutrition_req['Serat'],       "g"),
                ("Natrium",     nutrition_req['Natrium'],     "mg"),
                ("Kalium",      nutrition_req['Kalium'],      "mg"),
                ("Kalsium",     nutrition_req['Kalsium'],     "mg"),
                ("Magnesium",   nutrition_req['Magnesium'],   "mg"),
                ("Besi",        nutrition_req['Besi'],        "mg"),
                ("VitaminC",    nutrition_req['VitaminC'],    "mg"),
                ("VitaminD",    nutrition_req['VitaminD'],    "mcg"),
            ]
            cols = st.columns(4)
            for i, (nama, nilai, satuan) in enumerate(nutrisi_display):
                cols[i % 4].metric(nama, f"{nilai:.1f} {satuan}")

        # --- GENERATE MEAL PLAN ---
        st.header("🍽️ Menu Makanan")
        with st.spinner("Menyusun meal plan..."):
            meal_plan    = generate_meal_plan(
                nutrition_req=nutrition_req, df_preprocessed=df_preprocessed,
                df_original=df_original, scaler=scaler, nutrient_cols=nutrient_cols, days=7)
            meal_plan_df = meal_plan_to_df(meal_plan)
        daily_nutrient = meal_plan_df.groupby('Hari')[nutrient_cols_eval].sum()
        tabs = st.tabs([f"Hari {d['Hari']}" for d in meal_plan])
        for tab, day_plan in zip(tabs, meal_plan):
            with tab:
                target   = nutrition_req['Energi']
                total    = day_plan['Total_Kalori']
                in_range = (target * 0.90) <= total <= (target * 1.10)
                status   = "✅ Dalam rentang target" if in_range else "⚠️ Di luar rentang target"
                total_p = sum(f['Protein'] for f in day_plan['Makanan'])
                total_l = sum(f['Lemak'] for f in day_plan['Makanan'])
                total_k = sum(f['Karbohidrat'] for f in day_plan['Makanan'])
                total_s = sum(f['Serat'] for f in day_plan['Makanan'])
                pct_k   = (total_k*4)/total*100 if total > 0 else 0
                pct_l   = (total_l*9)/total*100 if total > 0 else 0
                pct_p   = (total_p*4)/total*100 if total > 0 else 0

                current_meal = None
                for food in day_plan['Makanan']:
                    if food['WaktuMakan'] != current_meal:
                        current_meal = food['WaktuMakan']
                        slot_cal = sum(f['Energi'] for f in day_plan['Makanan'] if f['WaktuMakan'] == current_meal)
                        st.markdown(f"#### 🕐 {current_meal} &nbsp;&nbsp; `{slot_cal:.0f} kkal`")
                    porsi     = food.get('Porsi', 100)
                    porsi_str = f"{porsi}g" + (f" (+{(porsi//100)-1} porsi tambahan)" if porsi > 100 else "")
                    with st.expander(f"🥘 {food['FoodName']}  —  {food['Energi']:.0f} kkal · {porsi_str}"):
                        st.caption(f"{food['FoodCategory']} · {food['FoodGroup']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Protein** : {food['Protein']:.1f} g")
                            st.write(f"**Lemak** : {food['Lemak']:.1f} g")
                            st.write(f"**Karbohidrat** : {food['Karbohidrat']:.1f} g")
                            st.write(f"**Serat** : {food['Serat']:.1f} g")
                        with col2:
                            st.write(f"**Natrium** : {food['Natrium']:.0f} mg")
                            st.write(f"**Kalium** : {food.get('Kalium', 0):.0f} mg")
                            st.write(f"**Kalsium** : {food.get('Kalsium', 0):.0f} mg")
                            st.write(f"**Magnesium** : {food.get('Magnesium', 0):.0f} mg")
                        with col3:
                            st.write(f"**Besi** : {food.get('Besi', 0):.1f} mg")
                            st.write(f"**Vitamin C** : {food.get('VitaminC', 0):.1f} mg")
                            st.write(f"**Vitamin D** : {food.get('VitaminD', 0):.1f} mcg")

                st.subheader("Ringkasan Nutrisi Hari Ini")
                diff      = total - target
                diff_pct  = diff / target * 100
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                col_stat1.metric("Total Kalori", f"{total:.0f} kkal",
                                 f"{diff:+.0f} kkal ({diff_pct:+.1f}%)",
                                 delta_color="normal" if in_range else "inverse")
                col_stat1.caption("✅ Dalam rentang target (90–110%)" if in_range else "⚠️ Di luar rentang target")
                col_stat2.metric("Protein",     f"{total_p:.1f} g", f"{pct_p:.1f}% kalori")
                col_stat3.metric("Lemak",       f"{total_l:.1f} g", f"{pct_l:.1f}% kalori")
                col_stat4.metric("Karbohidrat", f"{total_k:.1f} g", f"{pct_k:.1f}% kalori")

                st.markdown("---")
                hari        = day_plan['Hari']
                row         = daily_nutrient.loc[hari]
                actual      = [row[n] for n in nutrient_cols_eval]
                target_vals = [nutrient_targets[n] for n in nutrient_cols_eval]
                fig, ax = plt.subplots(figsize=(10, 4))
                x     = np.arange(len(nutrient_cols_eval))
                width = 0.35
                ax.bar(x - width/2, actual,      width, label='Rekomendasi', color='#3498db', edgecolor='white')
                ax.bar(x + width/2, target_vals, width, label='Target', color='#bdc3c7', edgecolor='white')
                ax.set_xticks(x)
                ax.set_xticklabels(nutrient_cols_eval, rotation=45, ha='right', fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(axis='y', alpha=0.3)
                ax.set_title(f'Pemenuhan Nutrisi Hari {hari} vs Target', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Tabel pemenuhan nutrisi
                st.markdown("**Pemenuhan Nutrisi Hari Ini**")
                st.caption("Rentang optimal: Energi, Lemak, Karbohidrat, Protein, Serat (90–110%) · Natrium (≤100%)")
                pemenuhan_rows = []
                for col in nutrient_cols_eval:
                    tgt_val = nutrient_targets[col]
                    act_val = row[col] if col in row else 0
                    pct_val = act_val / tgt_val * 100 if tgt_val > 0 else 0

                    if col in ('Natrium'):
                        status_txt = '✅ Optimal' if pct_val <= 100 else '⚠️ Melebihi target'
                    else:
                        if 90 <= pct_val <= 110:
                            status_txt = '✅ Optimal'
                        elif pct_val < 90:
                            status_txt = '⚠️ Di bawah target'
                        else:
                            status_txt = '⚠️ Melebihi target'

                    pemenuhan_rows.append({
                        "Nutrisi"      : col,
                        "Rekomendasi"  : f"{act_val:.1f}",
                        "Target"       : f"{tgt_val:.1f}",
                        "Pemenuhan"    : f"{pct_val:.1f}%",
                        "Status"       : status_txt,
                    })
                st.dataframe(pd.DataFrame(pemenuhan_rows), use_container_width=True, hide_index=True)

        # --- EVALUASI ---
        st.header("📈 Evaluasi Meal Plan")
        total_per_hari = meal_plan_df.groupby('Hari')['Energi'].sum()
        target_kalori  = nutrition_req['Energi']
        mean_dist      = meal_plan_df['EuclideanDistance'].mean()

        # MAPE kalori
        mape_kalori = (total_per_hari - target_kalori).abs().mean() / target_kalori * 100

        # MAPE prediksi penurunan berat badan
        target_turun   = wlp['total_loss_target']
        mape_penurunan = abs(prediksi_turun - target_turun) / target_turun * 100

        # MAPE per nutrisi
        mape_eval = {}
        for col, tgt in nutrient_targets.items():
            if col in daily_nutrient.columns:
                mape_val       = (daily_nutrient[col] - tgt).abs().mean()
                mape_eval[col] = round(mape_val / tgt * 100, 1)
        rata_mape = sum(mape_eval.values()) / len(mape_eval)

        # Metrik utama
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("MAPE Kalori", f"{mape_kalori:.1f}%",
                   help="Rata-rata persentase selisih kalori harian vs target")
        ec2.metric("MAPE Penurunan", f"{mape_penurunan:.2f}%",
                   help="Persentase selisih prediksi penurunan vs target penurunan")
        ec3.metric("Rata-rata MAPE Nutrisi", f"{rata_mape:.1f}%",
                   help="Rata-rata MAPE dari semua nutrisi")
        ec4.metric("Mean Euclidean Distance", f"{mean_dist:.4f}",
                   help="Rata-rata jarak KNN semua makanan terpilih")

        # MAPE per nutrisi tabel
        st.subheader("MAPE per Nutrisi (% dari Target)")
        mape_rows = []
        for col, pct in mape_eval.items():
            mape_rows.append({"Nutrisi": col, "MAPE (%)": f"{pct:.1f}"})
        st.dataframe(pd.DataFrame(mape_rows), use_container_width=True, hide_index=True)

        # --- TABEL LENGKAP & DOWNLOAD ---
        with st.expander("📋 Tabel Lengkap Meal Plan 7 Hari"):
            show_cols = ['Hari','WaktuMakan','FoodGroup','FoodCategory',
                         'FoodName','Porsi','Energi','Protein','Lemak',
                         'Karbohidrat','Serat','Natrium',
                         'Kalium','Kalsium','Magnesium','Besi','VitaminC','VitaminD',
                         'EuclideanDistance']
            available = [c for c in show_cols if c in meal_plan_df.columns]
            st.dataframe(meal_plan_df[available], use_container_width=True, hide_index=True)
            csv = meal_plan_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV", csv,
                "meal_plan_7hari.csv", "text/csv",
                use_container_width=True
            )

else:
    st.info("👈 Isi data pengguna di sidebar lalu klik **Generate Meal Plan** untuk memulai.")
    with st.expander("Panduan Tingkat Aktivitas"):
        st.markdown("""
<small>

**Sedentary (tidak aktif)**
- Menghabiskan sebagian besar waktu dengan duduk, seperti bekerja di kantor, mengemudi, atau berada di rumah.
- Aktivitas harian hanya berupa berjalan seperlunya di rumah atau tempat kerja.
- Hampir tidak pernah berolahraga atau hanya sesekali dalam sebulan.

**Lightly Active (sedikit aktif)**
- Melakukan olahraga ringan hingga sedang sebanyak 1-3 kali per minggu selama sekitar 30-60 menit.
- Contohnya berjalan cepat, bersepeda santai, atau yoga ringan.
- Aktivitas sehari-hari juga melibatkan cukup banyak berdiri dan berjalan.

**Moderately Active (cukup aktif)**
- Melakukan olahraga sedang secara rutin sekitar 6-7 kali per minggu dengan durasi lebih dari 60 menit.
- Contohnya berlari, berenang, senam aerobik, sepak bola, atau bulu tangkis.
- Aktivitas harian juga banyak melibatkan bergerak dan berdiri.

**Very Active (sangat aktif)**
- Melakukan olahraga berat setiap hari atau bahkan dua kali sehari.
- Contohnya lari jarak jauh, latihan angkat beban, senam intensif, atau olahraga tim.
- Pekerjaan sehari-hari juga memerlukan banyak gerakan dan aktivitas fisik yang cukup berat.

**Extra Active (sangat aktif sekali)**
- Melakukan aktivitas fisik atau olahraga berat dua kali atau lebih setiap hari.
- Umumnya dilakukan oleh atlet profesional atau pekerja dengan beban fisik sangat tinggi, seperti pekerja konstruksi.
- Sebagian besar waktu dihabiskan untuk aktivitas fisik intensif.

</small>
""", unsafe_allow_html=True)