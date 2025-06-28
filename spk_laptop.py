from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load dataset sekali saat aplikasi start
df_full = pd.read_csv('laptop_data_cleaned.csv')

@app.route('/')
def index():
    companies = sorted(df_full['Company'].unique())
    types = sorted(df_full['TypeName'].unique())
    return render_template('index.html', companies=companies, types=types)

@app.route('/get_types/<brand>')
def get_types(brand):
    if brand == "Semua":
        types = sorted(df_full['TypeName'].unique())
    else:
        types = sorted(df_full[df_full['Company'] == brand]['TypeName'].unique())
    return jsonify(types)

@app.route('/hasil', methods=['POST'])
def hasil():
    selected_brand = request.form.get('brand')
    selected_type = request.form.get('type')
    preference = request.form.get('preference')

    df = df_full.copy()

    # Filter berdasarkan input
    if selected_brand != "Semua":
        df = df[df['Company'] == selected_brand]
    if selected_type != "Semua":
        df = df[df['TypeName'] == selected_type]

    if df.empty:
        return render_template('hasil.html', hasil=[], message="Tidak ada laptop yang sesuai.")

    criteria = ['Price', 'Ram', 'Weight']

    # Bobot berdasarkan preferensi
    def get_weights(p):
        return {
            "1": {'Price': 0.6, 'Ram': 0.15, 'Weight': 0.25},
            "2": {'Price': 0.10, 'Ram': 0.6, 'Weight': 0.3},
            "3": {'Price': 0.15, 'Ram': 0.15, 'Weight': 0.7},
            "4": {'Price': 0.15, 'Ram': 0.15, 'Weight': 0.7},
            "5": {'Price': 0.4, 'Ram': 0.4, 'Weight': 0.2},
            "6": {'Price': 0.25, 'Ram': 0.25, 'Weight': 0.5},
        }.get(p, {})

    weights = get_weights(preference)

    # Normalisasi matriks keputusan
    matrix = df[criteria].astype(float).values
    norm_denominator = np.sqrt((matrix ** 2).sum(axis=0))
    norm_matrix = matrix / norm_denominator

    # Kalikan dengan bobot
    w = np.array([weights[c] for c in criteria])
    weighted_matrix = norm_matrix * w

    # Tipe kriteria cost/benefit
    criteria_type = {
        'Price': 'cost',
        'Ram': 'benefit',
        'Weight': 'cost'
    }

    # Tentukan solusi ideal positif dan negatif
    ideal_positive = []
    ideal_negative = []
    for i, crit in enumerate(criteria):
        if criteria_type[crit] == 'benefit':
            ideal_positive.append(weighted_matrix[:, i].max())
            ideal_negative.append(weighted_matrix[:, i].min())
        else:
            ideal_positive.append(weighted_matrix[:, i].min())
            ideal_negative.append(weighted_matrix[:, i].max())
    ideal_positive = np.array(ideal_positive)
    ideal_negative = np.array(ideal_negative)

    # Hitung jarak ke solusi ideal positif dan negatif
    dist_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    dist_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))

    # Hitung skor TOPSIS
    scores = dist_negative / (dist_positive + dist_negative)

    df['CPU'] = df['Cpu_brand'].fillna('Tidak tersedia')
    df['TOPSIS_Score'] = scores
    df['Dist_Positive'] = dist_positive
    df['Dist_Negative'] = dist_negative

    # Urutkan berdasarkan skor, ambil 10 teratas
    df_sorted = df.sort_values('TOPSIS_Score', ascending=False)
    top_laptops = df_sorted[['Company', 'TypeName', 'CPU', 'Gpu_brand', 'Ram', 'Price', 'Weight', 'TOPSIS_Score', 'Dist_Positive', 'Dist_Negative']].head(10).to_dict(orient='records')

    # Hitung skor terbaik di Python
    best_score = max(top_laptops, key=lambda x: x['TOPSIS_Score'])['TOPSIS_Score']

    return render_template('hasil.html', hasil=top_laptops, best_score=best_score, message=None)

if __name__ == '__main__':
    app.run(debug=True)
