import re
import os
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Bacarıq taksonomiyası və Sinonimlər (Qrup yoldaşınızın kodu)
# ---------------------------------------------------------------------------
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "sql", "r", "matlab",
    "react", "angular", "vue", "next.js", "node.js", "django", "flask",
    "spring", "laravel", "html", "css", "tailwind", "fastapi", "express",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data analysis", "pandas", "numpy", "tensorflow", "pytorch", "keras",
    "scikit-learn", "spark", "hadoop", "etl", "data visualization", "tableau",
    "power bi", "statistics", "a/b testing", "llm", "generative ai",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "linux", "git", "github actions", "serverless", "microservices",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
    "snowflake", "dynamodb",
    "agile", "scrum", "project management", "leadership", "communication",
    "stakeholder management", "mentoring", "jira",
    "cybersecurity", "penetration testing", "networking", "rest api",
    "graphql", "testing", "unit testing", "tdd",
]

SYNONYMS = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "postgres": "postgresql", "k8s": "kubernetes", "ml": "machine learning",
    "ai": "artificial intelligence", "cv": "computer vision",
    "ci cd": "ci/cd", "scikit learn": "scikit-learn",
    "g cloud": "gcp", "google cloud": "gcp",
}


def extract_skills(text: str) -> set:
    t = " " + text.lower().replace("\n", " ") + " "
    t = re.sub(r"[^a-z0-9+#/.\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    found = set()
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, t):
            found.add(skill)
    for raw, canon in SYNONYMS.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(raw) + r"(?![a-z0-9])", t):
            found.add(canon)
    return found


def semantic_score(cv_text: str, jd_text: str, model) -> float:
    emb = model.encode([cv_text, jd_text], normalize_embeddings=True)
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    return round(max(0.0, sim) * 100, 1)


def compute_match(cv_text: str, jd_text: str, model):
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)
    matched = sorted(cv_skills & jd_skills)
    missing = sorted(jd_skills - cv_skills)

    skill_recall = len(matched) / max(1, len(jd_skills))
    sem = semantic_score(cv_text, jd_text, model) / 100.0

    score = round(100 * (0.65 * sem + 0.35 * skill_recall), 1)
    return score, matched, missing, sem * 100, skill_recall * 100


def heuristic_suggestions(missing, sem_pct, skill_recall_pct):
    tips = []
    if missing:
        top = ", ".join(missing[:5])
        tips.append(f"**Bacarıq boşluğunu bağlayın.** İş elanında {top} qeyd olunub. Bu bacarıqları real təcrübəniz varsa CV-nizə əlavə edin.")
    if skill_recall_pct < 50:
        tips.append("**İş elanının lüğətini təkrarlayın.** ATS sistemləri açar sözləri hərfi şəkildə axtarır.")
    if sem_pct < 55:
        tips.append("**Relevantlığı artırın.** CV-niz bu rol üçün kifayət qədər fokuslu görünmür.")
    if not tips:
        tips.append("**Güclü uyğunluq!** Sözləri incə şəkildə dəyişin, nailiyyətləri rəqəmlərlə ifadə edin.")
    return tips


def llm_suggestions(cv_text, jd_text, score, matched, missing):
    try:
        import openai
        client = openai.OpenAI()
        prompt = f"..." # Qısaltdım, qrup yoldaşınızın promptunu eyni saxlayın
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

# --- YENİ ƏLAVƏ EDİLƏN FUNKSİYA: Namizədi bazaya yazmaq ---
def save_candidate_to_db(name, email, score, matched, missing):
    db_file = "candidates.json"
    candidate = {
        "id": str(datetime.now().timestamp()),
        "name": name,
        "email": email,
        "score": score,
        "matched": matched,
        "missing": missing,
        "status": "Gözləmədə"
    }
    
    if os.path.exists(db_file):
        with open(db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(candidate)
    
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ---------------------------------------------------------------------------
# Streamlit interfeysi
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CV ↔ İş Elanı Uyğunluğu", page_icon="📄", layout="wide")
st.title("📄 AI CV ↔ İş Elanı Uyğunluq Analizi")
st.caption("Hakaton demosu • oflayn embedding-lər + bacarıq taksonomiyası")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- YENİ ƏLAVƏ: Namizəd məlumatları forması ---
st.subheader("📝 Şəxsi Məlumatlar")
col_info1, col_info2 = st.columns(2)
with col_info1:
    candidate_name = st.text_input("Ad və Soyad")
with col_info2:
    candidate_email = st.text_input("E-poçt ünvanı")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🧑‍💼 Namizəd CV-si")
    cv_file = st.file_uploader("CV yüklə (txt)", type=["txt"], key="cv")
    cv_text = st.text_area("…və ya CV mətnini bura yapışdırın", height=260, placeholder="CV mətnini bura yapışdırın…")
    if cv_file:
        cv_text = cv_file.read().decode("utf-8", "ignore")

with col2:
    st.subheader("🏢 İş Elanı")
    jd_file = st.file_uploader("İş elanını yüklə (txt)", type=["txt"], key="jd")
    jd_text = st.text_area("…və ya iş elanını bura yapışdırın", height=260, placeholder="İş elanını bura yapışdırın…")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8", "ignore")

if st.button("🔍 Uyğunluğu Analiz Et və Göndər", type="primary", use_container_width=True):
    if not candidate_name or not candidate_email:
        st.warning("Zəhmət olmasa Adınızı və E-poçtunuzu daxil edin!")
        st.stop()
    if not cv_text.strip() or not jd_text.strip():
        st.warning("Zəhmət olmasa həm CV, həm də iş elanı daxil edin.")
        st.stop()

    with st.spinner("Analiz edilir…"):
        score, matched, missing, sem_pct, recall_pct = compute_match(cv_text, jd_text, model)
        
        # --- YENİ ƏLAVƏ: Nəticəni yaddaşa yazırıq ---
        save_candidate_to_db(candidate_name, candidate_email, score, matched, missing)

    c1, c2, c3 = st.columns(3)
    c1.metric("Uyğunluq Balı", f"{score} / 100")
    c2.metric("Semantik Oxşarlıq", f"{sem_pct:.0f}%")
    c3.metric("Bacarıq Əhatəsi", f"{recall_pct:.0f}%")

    st.progress(min(score, 100) / 100)
    
    st.success("Təbriklər! Müraciətiniz HR panelinə uğurla göndərildi.")

    a, b = st.columns(2)
    with a:
        st.success(f"✅ Uyğun bacarıqlar ({len(matched)})")
        st.write(", ".join(matched) if matched else "_Heç biri aşkarlanmadı_")
    with b:
        st.error(f"❌ Çatışmayan bacarıqlar ({len(missing)})")
        st.write(", ".join(missing) if missing else "_Yoxdur — əla əhatə!_")

    st.subheader("💡 Təkmilləşdirmə tövsiyələri")
    llm = llm_suggestions(cv_text, jd_text, score, matched, missing)
    if llm:
        st.markdown(llm)
    else:
        for tip in heuristic_suggestions(missing, sem_pct, recall_pct):
            st.markdown(f"- {tip}")
