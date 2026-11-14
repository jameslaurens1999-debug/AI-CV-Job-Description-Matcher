import streamlit as st
import json
import os

st.set_page_config(page_title="HR İdarəetmə Paneli", page_icon="👔", layout="wide")

# 1. Məlumat bazasını oxumaq (Hackathon üçün JSON kifayətdir)
def load_candidates():
    if os.path.exists("candidates.json"):
        with open("candidates.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

candidates = load_candidates()

st.title("👔 HR İdarəetmə Paneli - Kaskad Axtarış")
st.markdown("Bu paneldə müraciətlər avtomatik olaraq **Tam (80-100%)**, **Qismən (50-79%)** və **Uyğun olmayan (<50%)** olaraq bölünür.")

if not candidates:
    st.info("Hələ ki, sistemdə heç bir namizəd yoxdur.")
    st.stop()

# 2. Namizədləri qruplara bölmək
full_match = [c for c in candidates if c['score'] >= 80]
partial_match = [c for c in candidates if 50 <= c['score'] < 80]
no_match = [c for c in candidates if c['score'] < 50]

# Ən yüksək baldan aşağıya doğru sıralamaq
full_match = sorted(full_match, key=lambda x: x['score'], reverse=True)
partial_match = sorted(partial_match, key=lambda x: x['score'], reverse=True)
no_match = sorted(no_match, key=lambda x: x['score'], reverse=True)

# 3. Kaskad Axtarış Parametrləri (HR seçir)
st.sidebar.header("Axtarış Tənzimləmələri")
required_count = st.sidebar.number_input("Vakansiya üçün neçə nəfər lazımdır?", min_value=1, value=3)

# 4. Kaskad Alqoritmi (Tam uyğunlar çatmırsa, qismən uyğunlardan götürür)
final_selection = []
if len(full_match) >= required_count:
    final_selection = full_match[:required_count]
    st.sidebar.success(f"Kifayət qədər Tam Uyğun namizəd tapıldı! ({len(full_match)} nəfər var)")
else:
    final_selection = full_match
    needed_more = required_count - len(full_match)
    final_selection.extend(partial_match[:needed_more])
    st.sidebar.warning(f"Tam uyğunlar azdır ({len(full_match)} nəfər). AI qismən uyğunları da siyahıya əlavə etdi!")

# 5. Seçilmiş Namizədlərin Ekranda Göstərilməsi
st.subheader(f"Nəticə: Ən uyğun {len(final_selection)} namizəd")

for idx, c in enumerate(final_selection):
    # Uyğunluq kateqoriyasını təyin edirik
    if c['score'] >= 80:
        badge = "🟢 Tam Uyğun"
    elif c['score'] >= 50:
        badge = "🟡 Qismən Uyğun"
    else:
        badge = "🔴 Uyğun Deyil"

    with st.expander(f"👤 {c['name']} | Uyğunluq: {c['score']}% | {badge}", expanded=(idx==0)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"📧 **E-poçt:** {c['email']}")
            st.success(f"✅ **Uyğun Bacarıqlar:** {', '.join(c['matched']) if c['matched'] else 'Yoxdur'}")
            st.error(f"❌ **Çatışmayanlar:** {', '.join(c['missing']) if c['missing'] else 'Yoxdur'}")
        
        with col2:
            st.metric("Süni İntellekt Balı", f"{c['score']} / 100")
            
            # 1 Kliklə Dəvət düyməsi
            if st.button(f"✉️ {c['name']} üçün Görüş Dəvəti Göndər", key=f"email_{c['id']}"):
                # Burada əvvəlki addımda yazdığımız smtplib funksiyası işə düşə bilər.
                # Hackathon demosu üçün vizual bildiriş:
                st.toast(f"{c['email']} ünvanına dəvət maili göndərildi!", icon="✅")
                st.success("Dəvət uğurla göndərildi!")
                
st.markdown("---")
with st.expander("📊 Bütün Baza Statistikasını Gör (HR Analitikası)"):
    st.write(f"- **Tam uyğun:** {len(full_match)} nəfər")
    st.write(f"- **Qismən uyğun:** {len(partial_match)} nəfər")
    st.write(f"- **Uyğun olmayan:** {len(no_match)} nəfər")
