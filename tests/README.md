# 🧪 Automated Test Suite — COVID-19 AI

هذا المجلد يحتوي على حزمة الاختبارات الآلية الشاملة لمشروع **COVID-19 Cases Prediction & Analytics Dashboard**.

---

## 📁 هيكل الاختبارات (Test Suite Architecture)

```text
tests/
├── conftest.py                # إعدادات pytest والمحاكاة والبيانات الاصطناعية (Fixtures)
├── test_imports.py            # فحص استيراد التبعيات والبيئة البرمجية وإصدار Python
├── test_data_loading.py       # اختبارات تحميل البيانات ومعالجة المسارات والملفات المفقودة
├── test_data_validation.py    # التحقق من صحة واكتمال البيانات الوبائية وعدم وجود قيم سالبة
├── test_preprocessing.py      # اختبارات تنظيف البيانات، القيم الشاذة، وهندسة الخصائص الزمنية
├── test_models.py             # اختبارات بناء وتهيئة نموذج Gradient Boosting Regressor
├── test_model_prediction.py   # اختبارات التنبؤ الفردي والدفعات والتحويل العكسي expm1
├── test_model_evaluation.py   # اختبارات مقاييس الأداء (MAE, RMSE, R2, Adjusted SMAPE)
├── test_visualization.py      # اختبارات بناء وتوليد مخططات Plotly و Matplotlib
├── test_application.py        # اختبارات التكامل للواجهة التفاعلية باستخدام Streamlit AppTest
├── test_regression.py         # اختبارات الانحدار للعيوب المكتشفة لمنع تكرارها مستقبلاً
├── test_security.py           # الفحص الأمني للأسرار البرمجية وخصوصية البيانات
└── README.md                  # دليل استخدام وتشغيل الاختبارات
```

---

## 🚀 كيفية تشغيل الاختبارات (Running the Tests)

### 1. تشغيل جميع الاختبارات بالتفصيل
```bash
python -m pytest tests/ -v
```

### 2. تشغيل الاختبارات مع قياس نسبة التغطية (Coverage)
```bash
python -m pytest tests/ --cov=main --cov-report=term-missing
```

### 3. تشغيل ملف اختبار محدد
```bash
python -m pytest tests/test_regression.py -v
```

---

## 🧩 البيانات المستخدمة في الاختبارات (Fixtures & Synthetic Data)
- تم عزل معظم اختبارات الوحدة باستخدام بيانات اصطناعية ومحاكاة (`sample_covid_df` و `sample_vaccine_df`) في `conftest.py` لضمان:
  1. سرعة تنفيذ فائقة (ثوانٍ معدودة).
  2. استقلالية تامة وعدم الاعتماد على الاتصال بالإنترنت.
  3. حماية البيانات الأصلية من التعديل أو التلف.
