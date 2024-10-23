import random
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError  # 使用 Keras 的 MSE 损失函数

# 创建Flask应用
app = Flask(__name__)

# --- 加载保存的模型 ---
word2vec_model = joblib.load('word2vec_model.pkl')  # 加载Word2Vec模型
scaler = joblib.load('scaler.pkl')  # 加载Scaler
kmeans_model = joblib.load('kmeans_model.pkl')  # 加载KMeans模型

# 使用 custom_objects 来显式指定 'mse'
matrix_factorization_model = load_model('matrix_factorization_model.h5', custom_objects={'mse': MeanSquaredError()})

# 加载干净的车辆数据
df = pd.read_csv("clean_data.csv")

# 如果 'carId' 列不存在，创建该列
if 'carId' not in df.columns:
    df['carId'] = df.index

# 特征列
features = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'condition', 'color', 'interior', 'mmr']

# 初始化编码映射
car_ids = df['carId'].unique().tolist()
car2car_encoded = {x: i for i, x in enumerate(car_ids)}
car_encoded2car = {i: x for i, x in enumerate(car_ids)}

# 数据预处理：Word2Vec嵌入和标准化
df['carId'] = df.index
car_vectors = df.apply(lambda car: np.concatenate([
    np.mean([word2vec_model.wv[feature] for feature in car[features[:-1]].values.astype(str) if feature in word2vec_model.wv] or [np.zeros(100)], axis=0),
    np.array([car['mmr'] if pd.notna(car['mmr']) else 0])  # 手动添加 mmr 特征，确保没有 NaN
]), axis=1).tolist()

# 现在进行标准化
car_vectors = scaler.transform(car_vectors)

# 动态过滤用户提供的有效输入
def apply_filters(filtered_df, user_input, min_price, max_price, min_odo, max_odo):
    try:
        # 处理数值区间（价格和里程）
        if min_price > -np.inf or max_price < np.inf:
            filtered_df = filtered_df[(filtered_df['sellingprice'] >= min_price) & (filtered_df['sellingprice'] <= max_price)]
        if min_odo > -np.inf or max_odo < np.inf:
            filtered_df = filtered_df[(filtered_df['odometer'] >= min_odo) & (filtered_df['odometer'] <= max_odo)]
        
        return filtered_df
    except Exception as e:
        print(f"Error in apply_filters: {e}")
        return pd.DataFrame()

# 基于内容的推荐函数
def content_based_recommendation(user_input, top_n=10, min_price=-np.inf, max_price=np.inf, min_odo=-np.inf, max_odo=np.inf):
    try:
        # 初始未过滤的车辆数据
        filtered_df = df.copy()

        # 仅根据用户提供的有效输入进行过滤
        filtered_df = apply_filters(filtered_df, user_input, min_price, max_price, min_odo, max_odo)

        if filtered_df.empty:
            return pd.DataFrame()  # 如果没有符合条件的车辆，返回空数据

        # 构建用户向量，考虑 mmr 特征
        user_vector = np.concatenate([
            np.mean([word2vec_model.wv[feature] for feature in user_input.values() if feature in word2vec_model.wv] or [np.zeros(100)], axis=0),
            np.array([user_input.get('mmr', 0)])  # 手动添加 mmr 特征，确保向量长度为 101
        ])
        user_vector = scaler.transform([user_vector])

        # 计算相似度
        similarities = cosine_similarity(user_vector, car_vectors[filtered_df.index])[0]
        similar_car_indices = np.argsort(similarities)[::-1][:top_n]

        # 返回推荐的车辆
        return filtered_df.iloc[similar_car_indices]
    except Exception as e:
        print(f"Error in content_based_recommendation: {e}")
        return pd.DataFrame()

# 空推荐函数：随机选择车辆
def empty_recommendation(top_n=10):
    try:
        return df.sample(n=top_n)
    except Exception as e:
        print(f"Error in empty_recommendation: {e}")
        return pd.DataFrame()

# 矩阵分解推荐函数，使用 H5 模型
def matrix_factorization_recommendation(user_ratings, user_id, top_n=10):
    try:
        if user_ratings.empty:
            return pd.DataFrame()  # 如果用户评分为空，返回空结果

        # 初始化用户ID编码映射
        user_ids = user_ratings['userId'].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}

        cars_not_watched = df[~df["carId"].isin(user_ratings['carId'])]["carId"]
        cars_not_watched_encoded = [car2car_encoded.get(x) for x in cars_not_watched if x in car2car_encoded]
        
        user_encoded = user2user_encoded.get(user_id)
        if user_encoded is None:
            return pd.DataFrame()

        user_car_array = np.hstack(([[user_encoded]] * len(cars_not_watched_encoded), np.array(cars_not_watched_encoded).reshape(-1, 1)))

        predictions = matrix_factorization_model.predict([user_car_array[:, 0], user_car_array[:, 1]]).flatten()
        top_rated_indices = predictions.argsort()[-top_n:][::-1]

        recommended_car_ids = [car_encoded2car.get(cars_not_watched_encoded[x]) for x in top_rated_indices]
        recommended_cars = df[df["carId"].isin(recommended_car_ids)]
        return recommended_cars
    except Exception as e:
        print(f"Error in matrix_factorization_recommendation: {e}")
        return pd.DataFrame()
    
def predict_rating(model, user_id, car_id, user2user_encoded, car2car_encoded, mean_rating):
    """使用矩阵分解模型预测用户对车辆的评分."""
    user_encoded = user2user_encoded.get(user_id)
    car_encoded = car2car_encoded.get(car_id)
    if user_encoded is not None and car_encoded is not None:
        rating = model.predict([np.array([user_encoded]), np.array([car_encoded])])
        return rating[0][0]
    else:
        return mean_rating  # 使用平均评分作为默认值

# Flask路由用于处理推荐请求
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # 获取用户输入数据
        search_data = request.get_json()
        print("Received search data: ", search_data)

        # 构建用户输入字典，并处理空值
        user_input = {
            'make': search_data.get('make', ''),
            'model': search_data.get('model', ''),
            'trim': search_data.get('trim', ''),
            'body': search_data.get('body', ''),
            'transmission': search_data.get('transmission', ''),
            'state': search_data.get('state', ''),
            'condition': search_data.get('condition', ''),
            'color': search_data.get('color', ''),
            'interior': search_data.get('interior', ''),
            'mmr': float(search_data.get('mmr', 0)) if search_data.get('mmr') else 0  # 处理 mmr 字段，转换为 float 或 0
        }

        # 获取价格和里程数过滤条件，处理空字符串，确保空字符串不会被转换为 float
        min_price = float(search_data['minPrice']) if search_data.get('minPrice') not in [None, ''] else -np.inf
        max_price = float(search_data['maxPrice']) if search_data.get('maxPrice') not in [None, ''] else np.inf
        min_odo = float(search_data['minOdometer']) if search_data.get('minOdometer') not in [None, ''] else -np.inf
        max_odo = float(search_data['maxOdometer']) if search_data.get('maxOdometer') not in [None, ''] else np.inf

        # 将 recommendations 初始化为空 DataFrame
        recommendations = pd.DataFrame()

        # 基于内容的推荐（始终计算）
        if not user_input or not any(user_input.values()):
            content_recs = empty_recommendation()  # 如果没有输入，则回退到随机推荐
        else:
            content_recs = content_based_recommendation(user_input, top_n=10, min_price=min_price, max_price=max_price, min_odo=min_odo, max_odo=max_odo)

        # 获取用户评分（如果提供）
        has_ratings = 'user_ratings' in search_data and search_data['user_ratings']
        if has_ratings:
            user_ratings_df = pd.DataFrame(search_data['user_ratings'])
            user_id = search_data.get('user_id', 0)

            # 矩阵分解推荐（如果提供评分，则计算）
            matrix_recs = matrix_factorization_recommendation(user_ratings_df, user_id)

            # 如果基于内容的推荐和矩阵分解推荐都可用，则进行混合
            if not matrix_recs.empty and not content_recs.empty:
                # --- 混合推荐逻辑 ---

                # 1. 计算基于内容的评分
                random_number = random.randint(500, 40000)
                user_vector = np.concatenate([
                    np.mean([word2vec_model.wv[feature] for feature in user_input.values() if feature in word2vec_model.wv] or [np.zeros(100)], axis=0),
                    np.array([user_input.get('mmr', random_number)])  # 包含 MMR（如果可用）
                ])
                user_vector = scaler.transform([user_vector])
                content_recs['content_score'] = content_recs.apply(
                    lambda row: cosine_similarity(user_vector, [car_vectors[row.name]])[0][0], axis=1
                )

                # 2. 计算矩阵分解评分
                user_ids = user_ratings_df['userId'].unique().tolist()
                user2user_encoded = {x: i for i, x in enumerate(user_ids)}
                mean_rating = user_ratings_df['rating'].mean()

                content_recs['matrix_score'] = content_recs.apply(
                    lambda row: predict_rating(matrix_factorization_model, user_id, row.name, user2user_encoded, mean_rating),
                    axis=1
                )

                # 3. 结合评分（根据需要调整权重）
                content_recs['combined_score'] = 0.5 * content_recs['content_score'] + 0.5 * content_recs['matrix_score']

                # 4. 按组合评分排序并获取前 10 个推荐
                recommendations = content_recs.sort_values(by='combined_score', ascending=False).head(10)
            elif not matrix_recs.empty:  # 用户已评分，但未提供其他条件，则仅使用矩阵分解
                recommendations = matrix_recs
        else:  # 没有评分，则仅使用基于内容的过滤
            recommendations = content_recs

        # 返回推荐结果
        if recommendations.empty:
            return jsonify([]), 200  # 如果没有推荐，则返回空列表
        return jsonify(recommendations.to_dict(orient="records")), 200

    except Exception as e:
        print(f"Error in /recommend route: {e}")
        return jsonify({"error": str(e)}), 500

# 启动Flask服务
if __name__ == "__main__":
    app.run(port=5000, debug=True)
