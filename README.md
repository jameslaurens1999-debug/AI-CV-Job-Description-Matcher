# 📄 AI CV ↔ İş Elanı Uyğunluq Analizi

Hakaton demosu: CV-ni və iş elanını yükləyin — dərhal **uyğunluq balı**,
bacarıq boşluğu analizi və praktik təkmilləşdirmə tövsiyələri əldə edin.

![demo](https://img.shields.io/badge/streamlit-app-orange) ![offline](https://img.shields.io/badge/offline-i%C5%9Fl%C9%99yir-brightgreen)

## ✨ Xüsusiyyətlər

- **Uyğunluq balı (0–100)** — semantik oxşarlıq (sentence-transformers) + bacarıq əhatəsinin qarışığı
- **Bacarıq çıxarılması** — sinonim dəstəyi olan ~80 texniki/yumşaq bacarıq taksonomiyası
- **Boşluq analizi** — uyğun və çatışmayan bacarıqlar yan-yana
- **Təkmilləşdirmə tövsiyələri** — standart olaraq qayda əsaslı, `OPENAI_API_KEY` təyin olunanda **LLM ilə**

## 🚀 Tez Başlanğıc

```bash
pip install -r requirements.txt
streamlit run app.py
