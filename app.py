import re
import os
from pathlib import Path
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Bacarıq taksonomiyası (sərbəst genişləndirə bilərsiniz!)
# ---------------------------------------------------------------------------
SKILLS = [
    # proqramlaşdırma dilləri
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "php", "ruby", "swift", "kotlin", "scala", "sql", "r", "matlab",
    # veb və freymvorklar
    "react", "angular", "vue", "next.js", "node.js", "django", "flask",
    "spring", "laravel", "html", "css", "tailwind", "fastapi", "express",
    # data və ml
    "machine learning", "deep learning", "nlp", "computer vision",
    "data analysis", "pandas", "numpy", "tensorflow", "pytorch", "keras",
    "scikit-learn", "spark", "hadoop", "etl", "data visualization", "tableau",
    "power bi", "statistics", "a/b testing", "llm", "generative ai",
    # devops və bulud
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "linux", "git", "github actions", "serverless", "microservices",
    # verilənlər bazaları
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
    "snowflake", "dynamodb",
    # idarəetmə və yumşaq bacarıqlar
    "agile", "scrum", "project management", "leadership", "communication",
    "stakeholder management", "mentoring", "jira",
    # təhlükəsizlik və digər
    "cybersecurity", "penetration testing", "networking", "rest api",
    "graphql", "testing", "unit testing", "tdd",
]

# Sinonimlər — müxtəlif yazılışları vahid adla əvəz edir
SYNONYMS = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "postgres": "postgresql", "k8s": "kubernetes", "ml": "machine learning",
    "ai": "artificial intelligence", "cv": "computer vision",
    "ci cd": "ci/cd", "scikit learn": "scikit-learn",
    "g cloud": "gcp", "google cloud": "gcp",
}


def extract_skills(text: str) -> set:
    """Bacarıq çıxarılması (kiçik hərflərə uyğun, sinonimlərlə)."""
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
    """CV və JD embedding-ləri arasında kosinus oxşarlığı -> 0-100."""
    emb = model.encode([cv_text, jd_text], normalize_embeddings=True)
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    return round(max(0.0, sim) * 100, 1)


def compute_match(cv_text: str, jd_text: str, model):
    """Bacarıq üst-üstə düşməsi + semantik oxşarlıq = yekun bal."""
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)
    matched = sorted(cv_skills & jd_skills)
    missing = sorted(jd_skills - cv_skills)

    skill_recall = len(matched) / max(1, len(jd_skills))          # 0..1
    sem = semantic_score(cv_text, jd_text, model) / 100.0        # 0..1

    # Çəkili qaynaşma: semantik oxşarlıq əsasdır, bacarıq əhatəsi isə
    # vacib tələblərin örtülməsini artırır.
    score = round(100 * (0.65 * sem + 0.35 * skill_recall), 1)
    return score, matched, missing, sem * 100, skill_recall * 100


def heuristic_suggestions(missing, sem_pct, skill_recall_pct):
    """LLM qoşulmayıbsa istifadə olunan qayda əsaslı tövsiyələr."""
    tips = []
    if missing:
        top = ", ".join(missing[:5])
        tips.append(
            f"**Bacarıq boşluğunu bağlayın.** İş elanında {top} qeyd olunub. "
            "Bu bacarıqları real təcrübəniz varsa CV-nizə əlavə edin "
            "(layihələr, kurslar və sertifikatlar da sayılır)."
        )
    if skill_recall_pct < 50:
        tips.append(
            "**İş elanının lüğətini təkrarlayın.** ATS sistemləri açar sözləri "
            "hərfi şəkildə axtarır — elanda 'CI/CD pipelines' yazılıbsa, "
            "CV-də yalnız 'avtomatlaşdırılmış deploy' yazmayın."
        )
    if sem_pct < 55:
        tips.append(
            "**Relevantlığı artırın.** CV-niz bu rol üçün kifayət qədər "
            "fokuslu görünmür. Ən uyğun təcrübəni yuxarı çıxarın və "
            "nəticələri rəqəmlərlə göstərin (məs., 'gecikməni 40% azaltdım')."
        )
    if not tips:
        tips.append(
            "**Güclü uyğunluq!** Sözləri incə şəkildə dəyişin, nailiyyətləri "
            "rəqəmlərlə ifadə edin və daha çox elan ifadəsini təkrarlayın."
        )
    tips.append(
        "_Məsləhət: cümlə-səviyyəli CV yenidən yazıları üçün `.env` faylında "
        "`OPENAI_API_KEY` təyin edin._"
    )
    return tips


def llm_suggestions(cv_text, jd_text, score, matched, missing):
    """İstəyə bağlı LLM tövsiyələri (OpenAI-uyğun). Təhlükəsiz fallback."""
    try:
        import openai
        client = openai.OpenAI()  # OPENAI_API_KEY env-dən oxunur
        prompt = f"""You are a career coach. A candidate has a {score}/100 match with this job.
Matched skills: {matched}
Missing skills: {missing}

JOB DESCRIPTION:
{jd_text[:4000]}

CV:
{cv_text[:4000]}

Give 3-5 concise, actionable suggestions in AZERBAIJANI to improve THIS cv for THIS job.
Be specific (which bullet to change and how). Markdown bullets, max 15 words each."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streamlit interfeysi
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CV ↔ İş Elanı Uyğunluğu", page_icon="📄", layout="wide")
st.title("📄 AI CV ↔ İş Elanı Uyğunluq Analizi")
st.caption("Hakaton demosu • oflayn embedding-lər + bacarıq taksonomiyası • istəyə bağlı LLM tövsiyələri")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🧑‍💼 Namizəd CV-si")
    cv_file = st.file_uploader("CV yüklə (txt)", type=["txt"], key="cv")
    cv_text = st.text_area("…və ya CV mətnini bura yapışdırın", height=260,
                           placeholder="CV mətnini bura yapışdırın…")
    if cv_file:
        cv_text = cv_file.read().decode("utf-8", "ignore")
    for p in Path("samples").glob("cv_*.txt"):
        if st.button(f"Nümunəyi sına: {p.name}", key=p.name):
            cv_text = p.read_text()

with col2:
    st.subheader("🏢 İş Elanı")
    jd_file = st.file_uploader("İş elanını yüklə (txt)", type=["txt"], key="jd")
    jd_text = st.text_area("…və ya iş elanını bura yapışdırın", height=260,
                           placeholder="İş elanını bura yapışdırın…")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8", "ignore")
    for p in Path("samples").glob("jd_*.txt"):
        if st.button(f"Nümunəyi sına: {p.name}", key=p.name):
            jd_text = p.read_text()

if st.button("🔍 Uyğunluğu Analiz Et", type="primary", use_container_width=True):
    if not cv_text.strip() or not jd_text.strip():
        st.warning("Zəhmət olmasa həm CV, həm də iş elanı daxil edin.")
        st.stop()

    with st.spinner("Analiz edilir…"):
        score, matched, missing, sem_pct, recall_pct = compute_match(
            cv_text, jd_text, model
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Uyğunluq Balı", f"{score} / 100")
    c2.metric("Semantik Oxşarlıq", f"{sem_pct:.0f}%")
    c3.metric("Bacarıq Əhatəsi", f"{recall_pct:.0f}%")

    st.progress(min(score, 100) / 100)

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
