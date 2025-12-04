#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[1]:


# --- 0. إعداد البيئة والمكتبات والمصادر (مُحدثة بالكامل) ---

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import ta # مكتبة المؤشرات الفنية
import yfinance as yf # المكتبة التي سنستخدمها الآن لجلب البيانات
import time # لإضافة تأخير بسيط بين المحاولات إذا لزم الأمر
from datetime import datetime

import warnings
warnings.filterwarnings('ignore') # لإلغاء التحذيرات

print("المكتبات الأساسية تم استيرادها بنجاح.")

# --- تحديد المسارات (نحن نعمل محليًا الآن) ---
OUTPUT_DIR = 'Stock_Prediction_Models_And_Results_Professional_Final_V4'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_models')
SCALERS_DIR = os.path.join(OUTPUT_DIR, 'trained_scalers')
DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
BACKTEST_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'backtesting_results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"سيتم حفظ جميع المخرجات في: {os.path.abspath(OUTPUT_DIR)}")

# --- المتغيرات والثوابت العامة ---
GLOBAL_STOCK_TICKER = 'NVDA'
START_DATE_DATA = '2000-01-01'
END_DATE_DATA = datetime.now().strftime('%Y-%m-%d')

FORECAST_HORIZON = 1
UP_THRESHOLD = 0.0
DOWN_THRESHOLD = 0.0

TRAIN_SIZE_RATIO = 0.70
VAL_SIZE_RATIO = 0.15
TEST_SIZE_RATIO = 0.15

if not (TRAIN_SIZE_RATIO + VAL_SIZE_RATIO + TEST_SIZE_RATIO == 1.0):
    print("تحذير: مجموع نسب تقسيم البيانات لا يساوي 1.0. قد يؤدي ذلك إلى أخطاء.")

print(f"تم تعيين FORECAST_HORIZON = {FORECAST_HORIZON} يوم.")
print(f"تم تعيين UP_THRESHOLD = {UP_THRESHOLD} و DOWN_THRESHOLD = {DOWN_THRESHOLD} لتعريف الهدف الثنائي.")


# --- دالة لجلب البيانات باستخدام yfinance (النسخة النهائية والمستقرة) ---
def fetch_data_yfinance(ticker, start_date, end_date):
    print(f"\nجاري محاولة جلب بيانات {ticker} من Yahoo Finance باستخدام yfinance...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"تحذير: yfinance أعاد DataFrame فارغًا لـ {ticker} للفترة {start_date} إلى {end_date}.")
            return None
        else:
            print(f"تم جلب بيانات {ticker} بنجاح باستخدام yfinance. عدد الصفوف: {len(df)}")
            df.index.name = 'Date'
            df.sort_index(inplace=True)
            
            # 🚨 الحل: معالجة أسماء الأعمدة لتصبح نصًا بسيطًا
            if isinstance(df.columns, pd.MultiIndex):
                # إذا كانت الأعمدة متعددة، نأخذ القيمة الأدنى
                df.columns = [col[0].lower().replace(' ', '_') for col in df.columns]
            else:
                # إذا كانت الأعمدة عادية، نبسطها كالمعتاد
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols_lower):
                print(f"خطأ: الأعمدة المطلوبة {required_cols_lower} ليست موجودة في بيانات yfinance بعد تبسيط الأسماء.")
                print(f"الأعمدة المتاحة بعد المعالجة: {df.columns.tolist()}")
                return None
            
            df = df[required_cols_lower]
            
            print("\n--- عينة من البيانات المحملة (أول 5 صفوف) بأسماء أعمدة موحدة: ---")
            print(df.head())
            return df
    except Exception as e:
        print(f"حدث خطأ أثناء جلب البيانات باستخدام yfinance لـ {ticker}: {e}")
        return None

# --- تنفيذ جلب البيانات ---
df_stock = None
data_file_path = os.path.join(DATA_DIR, f'{GLOBAL_STOCK_TICKER}_Historical_Data.csv')

if os.path.exists(data_file_path) and os.path.getsize(data_file_path) > 0:
    print(f"\nجاري محاولة تحميل البيانات من الملف المحلي: {data_file_path}...")
    try:
        df_stock = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
        df_stock.columns = [col.lower().replace(' ', '_') for col in df_stock.columns]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_stock.columns for col in required_cols):
            print("تحذير: الملف المحلي لا يحتوي على كل الأعمدة المطلوبة بالأسماء الموحدة. سأحاول جلب البيانات مرة أخرى.")
            df_stock = None
        elif df_stock.empty or len(df_stock) < 100:
            print("تحذير: الملف المحلي فارغ أو يحتوي على بيانات قليلة. سأحاول جلب البيانات مرة أخرى.")
            df_stock = None
        else:
            print(f"تم تحميل البيانات من الملف المحلي بنجاح. عدد الصفوف: {len(df_stock)}")
            print("أول 5 صفوف من البيانات المحملة محليًا:")
            print(df_stock.head())
    except Exception as e:
        print(f"خطأ أثناء تحميل البيانات من الملف المحلي: {e}. سأحاول جلبها من yfinance.")
        df_stock = None

if df_stock is None or df_stock.empty:
    df_stock = fetch_data_yfinance(GLOBAL_STOCK_TICKER, START_DATE_DATA, END_DATE_DATA)
    if df_stock is not None and not df_stock.empty:
        df_stock.to_csv(data_file_path)
        print(f"تم حفظ البيانات الجديدة في: {data_file_path}")

if df_stock is None or df_stock.empty:
    print("\n!!! فشل جلب البيانات من أي مصدر. لا يمكن المتابعة بدون بيانات صالحة. !!!")
else:
    df_final = df_stock # 🚨 تم تعيين df_final هنا
    print(f"\n--- جلب البيانات الأساسية اكتمل بنجاح. جاهزون للخطوة التالية: هندسة الميزات المتقدمة. ---")


# In[2]:


# In[2]:


# --- 2. هندسة الميزات المتقدمة (Feature Engineering) ---

print("بدء عملية هندسة الميزات المتقدمة...")

# 🚨 تغيير اسم المتغير من df_spy إلى df_stock
if 'df_stock' not in locals() or df_stock is None or df_stock.empty:
    print("خطأ: DataFrame البيانات (df_stock) فارغ أو غير موجود. لا يمكن إجراء هندسة الميزات.")
    raise ValueError("البيانات الأساسية df_stock غير متاحة.")
else:
    df_temp = df_stock.copy()
    
    # 2.1 إنشاء المتغير المستهدف (Target Variable)
    if 'close' not in df_temp.columns:
        print("خطأ: العمود 'close' غير موجود. يرجى مراجعة الخلية السابقة.")
        raise KeyError("'close' column is missing.")
        
    df_temp['future_close'] = df_temp['close'].shift(-1)
    df_temp['future_return'] = (df_temp['future_close'] - df_temp['close']) / df_temp['close']
    
    def define_binary_target(row):
        return 1 if row['future_return'] > 0 else -1

    df_temp['target'] = df_temp.apply(define_binary_target, axis=1)
    df_temp['target'] = df_temp['target'].astype(int)
    
    df_temp.drop(columns=['future_close', 'future_return'], inplace=True)
    
    print("تم إنشاء المتغير المستهدف 'target' بنجاح.")

    # 2.2 إضافة مجموعة مختارة من المؤشرات الفنية (Selected Technical Indicators)
    df_temp['sma_20'] = ta.trend.sma_indicator(close=df_temp['close'], window=20)
    df_temp['rsi_14'] = ta.momentum.rsi(close=df_temp['close'], window=14)
    df_temp['atr_14'] = ta.volatility.average_true_range(high=df_temp['high'], low=df_temp['low'], close=df_temp['close'], window=14)
    
    print("تم إضافة مجموعة مختارة من المؤشرات الفنية.")

    # 2.3 إضافة الميزات الزمنية والموسمية (Time and Seasonal Features)
    df_temp['year'] = df_temp.index.year
    df_temp['month'] = df_temp.index.month
    df_temp['dayofweek'] = df_temp.index.dayofweek
    
    print("تم إضافة الميزات الزمنية والموسمية.")

    # 2.4 ميزات العوائد المتأخرة والتقلب (Lagged Returns & Volatility)
    for i in [1, 5, 20]:
        df_temp[f'lag_{i}_close_return'] = df_temp['close'].pct_change(periods=i).shift(1)
        df_temp[f'volatility_{i}_std'] = df_temp['close'].rolling(window=i).std()
    
    print("تم إضافة ميزات العوائد المتأخرة والتقلب.")
    
    # 2.5 ميزات الفجوة (Gap Features)
    df_temp['gap_open_close_prev_pct'] = ((df_temp['open'] - df_temp['close'].shift(1)) / df_temp['close'].shift(1)) * 100
    df_temp['daily_range_pct'] = ((df_temp['high'] - df_temp['low']) / df_temp['close']) * 100
    
    print("تم إضافة ميزات الفجوة.")
    
    # 2.6 تنظيف البيانات النهائية (Final Data Cleaning)
    initial_rows = len(df_temp)
    
    # بدلاً من حذف الصفوف، سنقوم بملء القيم المفقودة بالأصفار
    df_temp.fillna(0, inplace=True)
    
    # 🚨 تم إزالة هذه السطور لمنع الأخطاء الغير ضرورية
    # rows_filled = initial_rows - len(df_temp)
    # if rows_filled > 0:
    #     print(f"تم ملء {rows_filled} صفوف بقيم 0 بدلاً من حذفها.")
    
    # 2.7 تحديد الأعمدة النهائية للميزات والمتغير المستهدف
    features_to_drop = ['open', 'high', 'low', 'close', 'volume']
    X = df_temp.drop(columns=features_to_drop + ['target'], errors='ignore')
    y = df_temp['target']
    
    df_final = pd.concat([X, y], axis=1) # 🚨 تم تعيين df_final هنا

    print("\n--- هندسة الميزات المتقدمة مكتملة. ---")
    print(f"شكل البيانات النهائية: {df_final.shape}")


# In[3]:


# In[3]:

# --- 3.1 تقسيم البيانات (Split Data) ---
# سنقوم بتقسيم البيانات إلى مجموعات تدريب، وتحقق، واختبار
# مع الحفاظ على الترتيب الزمني لتجنب الانحياز المستقبلي (Look-Ahead Bias)

# تحديد تاريخ نهاية التدريب والتحقق
training_end_date = '2021-08-03'
validation_end_date = '2023-01-01'

# 🚨 تم تغيير اسم المتغير من df_spy_loaded إلى df_final
if 'df_final' not in locals() or df_final is None or df_final.empty:
    print("خطأ: البيانات النهائية df_final غير موجودة. يرجى التأكد من تشغيل الخلية السابقة بنجاح.")
    raise ValueError("البيانات النهائية غير متاحة.")
    
df_train = df_final.loc[df_final.index <= training_end_date]
df_val = df_final.loc[(df_final.index > training_end_date) & (df_final.index <= validation_end_date)]
df_test = df_final.loc[df_final.index > validation_end_date]

X_train, y_train = df_train.drop('target', axis=1), df_train['target']
X_val, y_val = df_val.drop('target', axis=1), df_val['target']
X_test, y_test = df_test.drop('target', axis=1), df_test['target']

print("--- تم تقسيم البيانات بنجاح ---")
print(f"شكل بيانات التدريب: {X_train.shape}, شكل المتغير المستهدف للتدريب: {y_train.shape}")
print(f"شكل بيانات التحقق: {X_val.shape}, شكل المتغير المستهدف للتحقق: {y_val.shape}")
print(f"شكل بيانات الاختبار: {X_test.shape}, شكل المتغير المستهدف للاختبار: {y_test.shape}")

# --- 3.2 معالجة عدم توازن الفئات (Oversampling) ---
print("\n--- معالجة عدم توازن الفئات باستخدام SMOTE ---")
# تحديد الفئات والنسبة
print(f"توزيع فئات التدريب الأصلي: {Counter(y_train)}")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"توزيع فئات التدريب بعد SMOTE: {Counter(y_train_res)}")

# --- 3.3 تحسين المعايير الفائقة باستخدام Optuna (Hyperparameter Tuning) ---
def objective(trial):
    """دالة الهدف لـ Optuna"""
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'early_stopping_round': 100
    }
    
    model = lgb.LGBMClassifier(**param)
    
    # تدريب النموذج على بيانات التدريب المعالجة واختباره على بيانات التحقق
    model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)])
    
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    return f1

print("\n--- بدء تحسين المعايير الفائقة باستخدام Optuna... ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000) # يمكن زيادة عدد المحاولات لنتائج أفضل

best_params = study.best_params
print("أفضل المعايير الفائقة التي تم العثور عليها:\n", best_params)

# تدريب النموذج النهائي باستخدام أفضل المعايير
print("\n--- تدريب النموذج النهائي بأفضل المعايير... ---")
final_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train_res, y_train_res)

# حفظ النموذج النهائي
joblib.dump(final_model, os.path.join(MODELS_DIR, 'best_lgbm_model.pkl'))
print(f"تم حفظ النموذج النهائي في: {os.path.join(MODELS_DIR, 'best_lgbm_model.pkl')}")
print("--- تدريب النموذج وتحسينه اكتمل بنجاح. ---")


# In[4]:


# In[4]:


# --- 4. التقييم الشامل وفهم النموذج ---

print("بدء عملية التقييم الشامل وفهم النموذج...")

# 🚨 التأكد من وجود النموذج والبيانات
if 'final_model' not in locals():
    print("خطأ: النموذج النهائي final_model غير موجود. يرجى التأكد من تشغيل الخلية السابقة بنجاح.")
    raise ValueError("النموذج النهائي غير متاح.")

if 'X_test' not in locals() or X_test is None or X_test.empty:
    print("خطأ: بيانات الاختبار X_test فارغة أو غير موجودة.")
    raise ValueError("بيانات الاختبار غير متاحة.")

# جاري التنبؤ على بيانات الاختبار...
y_pred = final_model.predict(X_test)
# احتمالات التنبؤ (مهمة خاصة لو كنا بنحلل أكثر من فئتين أو لتحليل الثقة)
y_pred_proba = final_model.predict_proba(X_test) 

# 4.2 تقييم أداء النموذج
print("\n--- تقييم أداء النموذج على بيانات الاختبار ---")

# حساب مقاييس الأداء
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro_test = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1-Score (Macro): {f1_macro_test:.4f}")

# تقرير التصنيف الشامل
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down (-1)', 'Up (1)'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
cm_df = pd.DataFrame(cm, index=['Actual Down', 'Actual Up'],
                     columns=['Predicted Down', 'Predicted Up'])
print("\nConfusion Matrix:")
print(cm_df)

# رسم Confusion Matrix بشكل مرئي
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False)
plt.title(f'Confusion Matrix for {GLOBAL_STOCK_TICKER} Stock Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plot_path_cm = os.path.join(PLOTS_DIR, f'{GLOBAL_STOCK_TICKER}_Confusion_Matrix.png')
plt.savefig(plot_path_cm)
plt.show()
print(f"تم حفظ Confusion Matrix في: {plot_path_cm}")

# 4.3 فهم أهمية الميزات (Feature Importance)
print("\n--- أهمية الميزات (Feature Importance) ---")

feature_importances = final_model.feature_importances_
feature_names = X_test.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("أهم 20 ميزة:")
print(importance_df.head(20))

# رسم أهمية الميزات
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
plt.title(f'Top 20 Feature Importances for {GLOBAL_STOCK_TICKER} Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plot_path_fi = os.path.join(PLOTS_DIR, f'{GLOBAL_STOCK_TICKER}_Feature_Importance.png')
plt.savefig(plot_path_fi)
plt.show()
print(f"تم حفظ رسم أهمية الميزات في: {plot_path_fi}")


print("\n--- التقييم الشامل وفهم النموذج اكتمل بنجاح. جاهزون للخطوة التالية: تحليل الحساسية والتفسير المتقدم (SHAP). ---")


# In[5]:


import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import warnings
import sys

warnings.filterwarnings('ignore')

# --- إعداد المسارات والمتغيرات العامة ---
DATA_DIR = 'Stock_Prediction_Models_And_Results_Professional_Final_V4/data'
MODELS_DIR = 'Stock_Prediction_Models_And_Results_Professional_Final_V4/models'
PLOTS_DIR = 'Stock_Prediction_Models_And_Results_Professional_Final_V4/plots'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

GLOBAL_STOCK_TICKER = 'NVDA'
# 💡 الحل: إضافة تعريف لمتغير RANDOM_STATE
RANDOM_STATE = 42

# مفاتيح API الخاصة بـ Alpaca Paper Trading
API_KEY = "PK3DNTKW56CXMXY46PWG"
API_SECRET = "0M0zj6pxqe35izpPf2yp05kxyBS4QiYEaufroKTZ"
BASE_URL = "https://paper-api.alpaca.markets"

# --- 1. تحميل ومعالجة البيانات الأولية من Alpaca API (مُحسّن) ---
print("⏳ جاري الاتصال بـ Alpaca API...")
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    print("✅ تم الاتصال بـ Alpaca API بنجاح.")
except Exception as e:
    print(f"❌ خطأ في الاتصال بـ Alpaca API: {e}")
    sys.exit()

print("🚨 جاري تحميل بيانات NVDA التاريخية من Alpaca...")
try:
    # تحديد نطاق زمني أكبر لتدريب النموذج
    end_date = datetime(2024, 1, 1)
    start_date = datetime(2010, 1, 1)
    barset = api.get_bars(
        GLOBAL_STOCK_TICKER,
        tradeapi.TimeFrame.Day,
        start=start_date.isoformat().split('T')[0], # تم إصلاح تنسيق التاريخ هنا
        end=end_date.isoformat().split('T')[0],   # وتم إصلاحه هنا
        limit=10000
    )
    df = barset.df
    if df.empty:
        raise ValueError("جدول البيانات فارغ بعد التحميل من Alpaca.")
    
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(None) # إزالة معلومات المنطقة الزمنية
    df.index.name = 'Date'
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    print("✅ تم تحميل البيانات من Alpaca بنجاح.")
    df_spy_loaded = df.copy()

except Exception as e:
    print(f"❌ حدث خطأ أثناء تحميل البيانات من Alpaca: {e}")
    sys.exit()

# --- 2. هندسة الميزات المتقدمة ---
df_spy_loaded['target'] = np.where(df_spy_loaded['close'].shift(-1) > df_spy_loaded['close'], 1, -1)
macd = MACD(close=df_spy_loaded['close'], window_fast=12, window_slow=26, window_sign=9)
df_spy_loaded['macd_line'] = macd.macd()
df_spy_loaded['macd_signal'] = macd.macd_signal()
bollinger = BollingerBands(close=df_spy_loaded['close'], window=20, window_dev=2)
df_spy_loaded['bb_hband'] = bollinger.bollinger_hband()
df_spy_loaded['bb_lband'] = bollinger.bollinger_lband()
df_spy_loaded['sma_20'] = SMAIndicator(close=df_spy_loaded['close'], window=20).sma_indicator()
df_spy_loaded['rsi_14'] = RSIIndicator(close=df_spy_loaded['close'], window=14).rsi()
df_spy_loaded['atr_14'] = df_spy_loaded['high'].rolling(14).max() - df_spy_loaded['low'].rolling(14).min()
df_spy_loaded['daily_range_pct'] = (df_spy_loaded['high'] - df_spy_loaded['low']) / df_spy_loaded['close'] * 100
df_spy_loaded['volatility_5_std'] = df_spy_loaded['close'].rolling(window=5).std()
df_spy_loaded['volatility_20_std'] = df_spy_loaded['close'].rolling(window=20).std()
df_spy_loaded['year'] = df_spy_loaded.index.year
df_spy_loaded['month'] = df_spy_loaded.index.month
df_spy_loaded['dayofweek'] = df_spy_loaded.index.dayofweek
for lag in [1, 5, 20]:
    df_spy_loaded[f'lag_{lag}_close_return'] = df_spy_loaded['close'].pct_change(lag) * 100
df_spy_loaded['gap_open_close_prev_pct'] = (df_spy_loaded['open'] - df_spy_loaded['close'].shift(1)) / df_spy_loaded['close'].shift(1) * 100
df_spy_loaded.dropna(inplace=True)

# --- 3. تحسين النموذج وتدريبه ---
df_train = df_spy_loaded.loc[df_spy_loaded.index <= '2021-08-03']
df_test = df_spy_loaded.loc[df_spy_loaded.index > '2021-08-03']

X_train, y_train = df_train.drop('target', axis=1), df_train['target']
X_test, y_test = df_test.drop('target', axis=1), df_test['target']

if X_train.empty:
    raise ValueError("بيانات التدريب فارغة. يرجى التحقق من تواريخ تقسيم البيانات.")

sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

best_params = {
    'n_estimators': 1200, 'learning_rate': 0.02, 'num_leaves': 64,
    'max_depth': 8, 'min_child_samples': 150, 'subsample': 0.8,
    'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.6
}
final_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
final_model.fit(X_train_res, y_train_res)
joblib.dump(final_model, os.path.join(MODELS_DIR, 'final_model.pkl'))

# --- 4. التقرير النهائي لأفضل أداء ---
print("بدء إنشاء التقرير النهائي لأداء النموذج...")
print("-" * 50)
y_pred_final = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_final)
f1_macro = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
print("## 📊 تقييم أداء النموذج")
print(f"✅ Accuracy (دقة التنبؤ): {accuracy:.4f}")
print(f"✅ F1-Score (مقياس شامل): {f1_macro:.4f}")
print("-" * 50)

# --- 5.1 إعداد المحاكاة التاريخية المُحسَّنة مع التكاليف ---
print("بدء عملية المحاكاة التاريخية المُحسَّنة لإيجاد أفضل قيم Stop-Loss و Take-Profit...")
sl_values = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 
 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029,
 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039,
 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049,
 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059,
 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069,
 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079,
 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089,
 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099,
 0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
 0.11, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
 0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129,
 0.13, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139,
 0.14, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149,
 0.15]

tp_values = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 
 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029,
 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039,
 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049,
 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059,
 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069,
 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079,
 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089,
 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099,
 0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109,
 0.11, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119,
 0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129,
 0.13, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139,
 0.14, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149,
 0.15]

TRANSACTION_COST_PERC = 0.001
SLIPPAGE_PERC = 0.0005

best_return = -np.inf
best_sl = 0
best_tp = 0
results_list = []
best_historical_capital = []

if 'final_model' not in locals() or 'X_test' not in locals() or 'df_test' not in locals():
    print("خطأ: النموذج أو بيانات الاختبار غير متوفرة. يرجى التأكد من تشغيل الخلايا السابقة بنجاح.")
    raise ValueError("الموارد غير متاحة.")

X_test_pred = X_test.copy()
X_test_pred['prediction'] = final_model.predict(X_test)
X_test_pred['close'] = df_test['close']
X_test_pred['actual_target'] = y_test

# --- 5.2 تنفيذ المحاكاة التاريخية (Backtesting) لجميع القيم الممكنة ---
for stop_loss_pct in sl_values:
    for take_profit_pct in tp_values:
        capital = 100000.00
        positions = {}
        current_historical_capital = []
        for date, row in X_test_pred.iterrows():
            prediction = row['prediction']
            current_close = row['close']
            if positions:
                for symbol, entry in list(positions.items()):
                    entry_price = entry['price']
                    num_shares = entry['shares']
                    current_profit_loss = (current_close - entry_price) / entry_price
                    if current_profit_loss <= -stop_loss_pct:
                        capital += num_shares * current_close
                        del positions[symbol]
                    elif current_profit_loss >= take_profit_pct:
                        capital += num_shares * current_close
                        del positions[symbol]
            if prediction == 1 and not positions:
                if capital > current_close:
                    num_shares = int(capital / current_close)
                    cost = num_shares * current_close
                    transaction_cost = cost * (TRANSACTION_COST_PERC + SLIPPAGE_PERC)
                    capital -= (cost + transaction_cost)
                    positions['SPY'] = {'shares': num_shares, 'price': current_close}
            current_historical_capital.append(capital + (positions['SPY']['shares'] * current_close if positions else 0))
        if positions:
            revenue = positions['SPY']['shares'] * X_test_pred['close'].iloc[-1]
            transaction_cost = revenue * (TRANSACTION_COST_PERC + SLIPPAGE_PERC)
            capital += (revenue - transaction_cost)
        strategy_return = (capital - 100000.00) / 100000.00 * 100
        results_list.append({
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct,
            'final_capital': capital,
            'strategy_return': strategy_return
        })
        if strategy_return > best_return:
            best_return = strategy_return
            best_sl = stop_loss_pct
            best_tp = take_profit_pct
            best_historical_capital = current_historical_capital

# --- 5.3 عرض النتائج النهائية ---
print("\n--- نتائج المحاكاة التاريخية المُحسَّنة ---")
print(f"أفضل عائد تم تحقيقه: {best_return:.2f}%")
print(f"أفضل قيم وقف الخسارة (Stop-Loss): {best_sl * 100:.2f}%")
print(f"أفضل قيم جني الأرباح (Take-Profit): {best_tp * 100:.2f}%")
buy_and_hold_return = (X_test_pred['close'].iloc[-1] - X_test_pred['close'].iloc[0]) / X_test_pred['close'].iloc[0] * 100
print(f"العائد الإجمالي لـ Buy and Hold: {buy_and_hold_return:.2f}%")
if best_historical_capital:
    df_capital = pd.Series(best_historical_capital)
    peak = df_capital.expanding(min_periods=1).max()
    drawdown = (df_capital - peak) / peak
    max_drawdown = drawdown.min() * 100
    print(f"أقصى تراجع (Max Drawdown): {max_drawdown:.2f}%")
else:
    max_drawdown = 0
print("\n--- المحاكاة التاريخية اكتملت. ---")

# رسم النتائج
df_historical_capital = pd.DataFrame(best_historical_capital, index=X_test_pred.index, columns=['Strategy'])
buy_and_hold_capital = (X_test_pred['close'] / X_test_pred['close'].iloc[0]) * 100000.00
df_historical_capital['Buy and Hold'] = buy_and_hold_capital

plt.figure(figsize=(12, 6))
plt.plot(df_historical_capital['Strategy'], label=f'Optimized Strategy (SL: {best_sl*100:.2f}%, TP: {best_tp*100:.2f}%)')
plt.plot(df_historical_capital['Buy and Hold'], label='Buy and Hold')
plt.title('Backtesting Results: Optimized Strategy vs. Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plot_path_optimized = os.path.join(PLOTS_DIR, 'Backtesting_Results_Optimized_Strategy_With_Costs.png')
plt.savefig(plot_path_optimized)
plt.show()
print(f"تم حفظ رسم المحاكاة التاريخية المُحسَّنة في: {plot_path_optimized}")


# In[ ]:


import pandas as pd
import numpy as np
import os
import joblib
import alpaca_trade_api as tradeapi
from ta.volatility import average_true_range, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from datetime import datetime, timedelta
import time
import pytz
import warnings

# Settings to ignore warnings
warnings.filterwarnings('ignore')

# --- Configuration and API Keys ---
MODELS_DIR = 'Stock_Prediction_Models_And_Results_Professional_Final_V4/models'
MODEL_FILENAME = 'final_model.pkl'
GLOBAL_STOCK_TICKER = 'NVDA' # This is now corrected to NVDA
RISK_PER_TRADE_PCT = 0.01

# Alpaca Paper Trading API keys (keep this information private)
API_KEY = "PK3DNTKW56CXMXY46PWG"
API_SECRET = "0M0zj6pxqe35izpPf2yp05kxyBS4QiYEaufroKTZ"
BASE_URL = "https://paper-api.alpaca.markets"

# The 24 feature names used to train the model
MODEL_FEATURE_NAMES = [
    'macd_line', 'macd_signal', 'macd_diff', 'bb_hband', 'bb_lband', 'bb_wband',
    'sma_20', 'rsi_14', 'atr_14', 'daily_range_pct', 'volatility_5_std',
    'volatility_20_std', 'year', 'month', 'dayofweek',
    'lag_1_close_return', 'lag_5_close_return', 'lag_20_close_return',
    'gap_open_close_prev_pct',
    'volume_change_pct', 'high_low_spread', 'open_close_spread', 'momentum_10d',
    'close_change_1d'
]

# --- 1. Load the pre-trained model ---
try:
    print("Loading the model...")
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    final_model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at the following path: {model_path}")
    exit()

# --- 2. Set up API connection ---
try:
    print("Connecting to Alpaca API...")
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"Successfully connected to Alpaca API.")
    print(f"Account status: {account.status}")
    print(f"Current portfolio value: ${account.equity}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")
    exit()

# --- 3. Function to check market open hours ---
def is_market_open():
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    if now.weekday() >= 5:
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now < market_close

# --- 4. Function to get data and prepare features (with retry mechanism) ---
def get_latest_features_with_retry(max_retries=5, delay_seconds=10):
    """
    Tries to fetch stock data with a retry mechanism to handle temporary connection issues.
    """
    for attempt in range(max_retries):
        try:
            # Getting minute-by-minute bars for the last 100 minutes
            barset = api.get_bars(GLOBAL_STOCK_TICKER, tradeapi.TimeFrame.Minute, limit=100).df
            
            if barset.empty or len(barset) < 50:
                 return None, None
            
            df = barset
            
            # Calculating all 24 features for the model
            macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd_line'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_hband'] = bollinger.bollinger_hband()
            df['bb_lband'] = bollinger.bollinger_lband()
            df['bb_wband'] = bollinger.bollinger_wband()
            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['atr_14'] = average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['volatility_5_std'] = df['close'].rolling(window=5).std()
            df['volatility_20_std'] = df['close'].rolling(window=20).std()
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['dayofweek'] = df.index.dayofweek
            for lag in [1, 5, 20]:
                df[f'lag_{lag}_close_return'] = df['close'].pct_change(lag) * 100
            df['gap_open_close_prev_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
            df['volume_change_pct'] = df['volume'].pct_change() * 100
            df['high_low_spread'] = df['high'] - df['low']
            df['open_close_spread'] = df['close'] - df['open']
            df['momentum_10d'] = df['close'].diff(10)
            df['close_change_1d'] = df['close'].pct_change(1)
            
            df.dropna(inplace=True)

            if df.empty:
                return None, None
                
            return df.iloc[-1], df
        except Exception as e:
            log_action(f"Error getting data from Alpaca (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                log_action(f"  - Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                log_action("  - Max retries reached. Giving up for this cycle.")
                return None, None

# --- New logging function ---
def log_action(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    with open('trading_log.txt', 'a') as f:
        f.write(f"{full_message}\n")
    print(full_message)

# --- 5. Main trading logic function (updated) ---
def trading_bot_logic():
    log_action("Checking the market...")
    if not is_market_open():
        log_action("  - Market is currently closed. Will re-check at the next scheduled time.")
        return

    try:
        # Use the new function with retry mechanism
        latest_data, full_df = get_latest_features_with_retry()
        if latest_data is None:
            log_action("  - No valid data received. Will try again later.")
            return

        current_price = latest_data['close']
        
        features_for_prediction_df = pd.DataFrame([latest_data])
        features_for_prediction_df = features_for_prediction_df[MODEL_FEATURE_NAMES]
        prediction = final_model.predict(features_for_prediction_df)[0]
        
        positions = api.list_positions()
        
        # Smart Buy Logic
        if not positions:
            if current_price > full_df['sma_50'].iloc[-1]:
                if prediction == 1:
                    sl_multiplier = 2.5
                    tp_multiplier = 5.0
                    stop_loss_amount = latest_data['atr_14'] * sl_multiplier
                    
                    account = api.get_account()
                    risk_per_trade_amount = float(account.equity) * RISK_PER_TRADE_PCT
                    
                    if stop_loss_amount > 0:
                        shares_to_buy = int(risk_per_trade_amount / stop_loss_amount)
                    else:
                        shares_to_buy = 0
                    
                    if shares_to_buy > 0:
                        api.submit_order(
                            symbol=GLOBAL_STOCK_TICKER,
                            qty=shares_to_buy,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        log_action(f"Buying {shares_to_buy} shares at ${current_price:.2f}.")
                        log_action(f"  - (Buy logic: Model recommended a buy and price is above SMA_50)")
                    else:
                        log_action("Capital or defined risk does not allow for a purchase at this time.")
                else:
                    log_action(f"  - Model does not recommend buying at the moment.")
            else:
                log_action(f"  - Current price (${current_price:.2f}) is below the moving average (SMA_50). Buy signal will be ignored.")
        
        # Smart Sell Logic
        else:
            position = positions[0]
            entry_price = float(position.avg_entry_price)
            
            sl_multiplier = 2.5
            tp_multiplier = 5.0
            atr_for_exit = latest_data['atr_14']
            stop_loss_price = entry_price - (atr_for_exit * sl_multiplier)
            take_profit_price = entry_price + (atr_for_exit * tp_multiplier)
            
            current_profit = (current_price - entry_price) / entry_price
            
            if current_price <= stop_loss_price:
                log_action(f"Closing trade (dynamic stop loss) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                api.close_position(GLOBAL_STOCK_TICKER)
            elif current_price >= take_profit_price:
                log_action(f"Closing trade (dynamic take profit) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                api.close_position(GLOBAL_STOCK_TICKER)
            elif latest_data['macd_line'] < latest_data['macd_signal']:
                log_action(f"Closing trade (negative MACD crossover) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                api.close_position(GLOBAL_STOCK_TICKER)
            elif latest_data['rsi_14'] > 70:
                log_action(f"Closing trade (RSI > 70) at ${current_price:.2f}. Return: {current_profit * 100:.2f}%")
                api.close_position(GLOBAL_STOCK_TICKER)
            else:
                log_action(f"  - Trade status: Current P/L: {current_profit * 100:.2f}%")

    except Exception as e:
        log_action(f"Error while executing trading logic: {e}")

# --- 6. Main execution loop (corrected) ---
log_action("--- Automated trading system started ---")
log_action("The code will now check the market every 5 minutes during market hours.")
log_action("To stop the code, press the stop button (red square) in Jupyter or Ctrl+C in the terminal.")
log_action("Starting the main execution loop.")

try:
    while True:
        trading_bot_logic()
        # Sleep for exactly 5 minutes (300 seconds)
        time.sleep(300)
except KeyboardInterrupt:
    log_action("--- Automated trading system stopped ---")


# In[ ]:




