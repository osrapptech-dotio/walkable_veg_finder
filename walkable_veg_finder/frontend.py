# ecoeats_mobile_app.py
import streamlit as st
import requests
import os
from datetime import datetime

# -------------------------
# Config
# -------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="EcoEats 🌱", layout="centered", initial_sidebar_state="collapsed")

# -------------------------
# CSS: Mobile styling
# -------------------------
st.markdown("""
<style>
body { background: linear-gradient(180deg, #eaf8ee 0%, #fbfff9 100%); }
.app-card {
  background: linear-gradient(180deg, #ffffffee, #ffffffcc);
  border-radius: 20px;
  padding: 22px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.08);
  margin: 16px auto;
  max-width: 420px;
  text-align: center;
}
.h1 { font-size: 32px; color: #1b5e20; font-weight: 800; margin-bottom: 6px; }
.h2 { font-size: 18px; color: #2e7d32; font-weight: 700; margin-bottom: 6px; }
.cta {
  width: 100%; height: 56px; border-radius: 14px; border: none;
  font-size: 17px; font-weight: 700; color: #fff;
  background: linear-gradient(90deg,#66bb6a,#43a047);
  box-shadow: 0 6px 16px rgba(67,160,71,0.18);
}
.ghost {
  width: 100%; height: 48px; border-radius: 12px;
  border: 1px solid #cde3cd; background: transparent;
  font-weight:700; color:#2e7d32;
}
.small-card {
  background:#ffffff; border-radius:12px; padding:10px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
  margin-bottom:10px; text-align:left;
}
.muted { color:#5b6b5b; font-size:13px; }
.badge { background: linear-gradient(90deg,#ffefc6,#e8f6e8); padding:6px 10px; border-radius:12px; font-weight:700; color:#2e7d32; }
.bottom-bar {
  position:fixed; left:0; right:0; bottom:0;
  background:#ffffffee; border-top-left-radius:12px; border-top-right-radius:12px;
  max-width:420px; margin:0 auto; padding:8px 10px;
  box-shadow:0 -6px 20px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session defaults
# -------------------------
defaults = {
    "screen": "welcome",
    "users": {},
    "auth": {"logged_in": False, "user": None},
    "points": 0,
    "_rerun_marker": 0,
    "_last_reco": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Helper: Safe rerun (kept for other navigation)
# -------------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state["_rerun_marker"] += 1
        st.stop()

# -------------------------
# Demo auth functions
# -------------------------
def signup(username, password):
    if username in st.session_state["users"]:
        return False, "User already exists"
    st.session_state["users"][username] = {"password": password, "points": 0, "joined": datetime.utcnow().isoformat()}
    return True, "Account created"

def login(username, password):
    u = st.session_state["users"].get(username)
    if not u:
        return False, "User not found"
    if u["password"] != password:
        return False, "Invalid password"
    st.session_state["auth"] = {"logged_in": True, "user": username}
    st.session_state["points"] = u.get("points", 0)
    return True, "Logged in"

def logout():
    st.session_state["auth"] = {"logged_in": False, "user": None}
    st.session_state["screen"] = "welcome"
    safe_rerun()

# -------------------------
# Forward declarations (functions used inline)
# -------------------------
def screen_preferences():
    """Render preferences screen (can be called directly after login to avoid needing a rerun)."""
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="h1">Your Preferences 🌾</div>', unsafe_allow_html=True)
    user = st.session_state["auth"].get("user") or "guest"
    st.markdown(f"<div class='muted'>Signed in as <b>{user}</b></div>", unsafe_allow_html=True)

    with st.form("prefs_form"):
        vegetarian = st.checkbox("Vegetarian", value=True)
        vegan = st.checkbox("Vegan", value=False)
        cuisines = st.text_input("Preferred cuisines (comma separated)", value="indian, italian")
        disliked = st.text_input("Disliked ingredients", value="")
        budget = st.number_input("Budget (USD)", min_value=0.0, value=20.0)
        st.markdown("---")
        st.markdown("♿ Accessibility")
        wheelchair = st.checkbox("Require wheelchair access", value=False)
        ramp = st.checkbox("Needs ramp", value=False)
        auto_door = st.checkbox("Needs automatic door", value=False)
        bath = st.checkbox("Needs accessible bathroom", value=False)
        max_walk = st.slider("Max walking distance (miles)", 0.5, 5.0, 2.0, 0.5)
        submitted = st.form_submit_button("💾 Save Preferences")
        if submitted:
            payload = {
                "user_id": user,
                "food": {
                    "vegetarian": vegetarian,
                    "vegan": vegan,
                    "cuisines": [c.strip() for c in cuisines.split(",") if c.strip()],
                    "disliked_ingredients": [d.strip() for d in disliked.split(",") if d.strip()],
                    "budget_usd": budget if budget > 0 else None,
                },
                "mobility": {
                    "wheelchair": wheelchair,
                    "needs_ramp": ramp,
                    "needs_auto_door": auto_door,
                    "needs_accessible_bathroom": bath,
                },
                "max_walk_miles": max_walk,
            }
            try:
                requests.post(f"{API_BASE}/onboarding", json=payload, timeout=6)
                st.success("Preferences saved ✅")
                st.session_state["screen"] = "recommend"
                safe_rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("⬅️ Back to Login"):
        st.session_state["screen"] = "login"
        safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def screen_recommend():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="h1">Find Nearby Restaurants 🍽️</div>', unsafe_allow_html=True)
    lat = st.number_input("Latitude", value=40.7429, format="%.6f")
    lon = st.number_input("Longitude", value=-73.9925, format="%.6f")

    if st.button("🔍 Get Recommendation"):
        req = {"user_id": st.session_state["auth"]["user"], "location": {"lat": float(lat), "lon": float(lon)}}
        try:
            res = requests.post(f"{API_BASE}/recommend", json=req, timeout=8)
            data = res.json()
        except Exception as e:
            st.error(f"API error: {e}")
            data = None
        if data and data.get("found"):
            r = data["result"]
            st.success(f"✅ {r['restaurant_name']}")
            st.markdown(f"**Distance:** {r['distance_miles']} miles")
            dish = r["best_dish"]
            st.markdown(f"**Dish:** {dish.get('name')} (${dish.get('price', '?')})")
            st.markdown(f"_{dish.get('description')}_")
            eco = round(max(0, 1.2 - (r['distance_miles'] / 2.0)), 3)
            st.metric("🌱 Eco Score", f"{eco}/1.0")
            st.session_state["_last_reco"] = r
        else:
            st.warning(data.get("reason", "No suitable restaurant found.") if data else "No data returned")

    if st.button("⬅️ Back to Preferences"):
        st.session_state["screen"] = "preferences"
        safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Screen 1: Welcome
# -------------------------
def screen_welcome():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/plant.png", width=88)
    st.markdown('<div class="h1">EcoEats 🌱</div>', unsafe_allow_html=True)
    st.markdown('<div class="h2">Find eco-friendly & vegetarian dining nearby</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Sustainable, accessible, and walkable food options.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button("🚀 Get Started", key="start"):
        st.session_state["screen"] = "login"
        safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Screen 2: Login / Sign Up
# -------------------------
def screen_login():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.image("https://img.icons8.com/ios-filled/100/leaf.png", width=72)
    st.markdown('<div class="h1">Login or Sign Up</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Login", "Sign up"], horizontal=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # LOGIN: on success, render preferences inline (avoids needing a rerun)
    if mode == "Login":
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                ok, msg = login(u.strip(), p)
                if ok:
                    st.success(msg)
                    # set auth state
                    st.session_state["auth"] = {"logged_in": True, "user": u.strip()}
                    st.session_state["points"] = st.session_state["users"].get(u.strip(), {}).get("points", 0)
                    # Directly render preferences immediately (no rerun required)
                    screen_preferences()
                    return
                else:
                    st.error(msg)
    # SIGNUP flow
    else:
        with st.form("signup_form"):
            u = st.text_input("Choose username")
            p = st.text_input("Choose password", type="password")
            if st.form_submit_button("Create account"):
                if not u or not p:
                    st.error("Please fill all fields")
                else:
                    ok, msg = signup(u.strip(), p)
                    if ok:
                        st.success(msg + " — you can now login.")
                    else:
                        st.error(msg)

    if st.button("⬅️ Back to Welcome"):
        st.session_state["screen"] = "welcome"
        safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Bottom navigation (simple)
# -------------------------
def bottom_nav():
    st.markdown(f"""
    <div class="bottom-bar">
      <div style="display:flex; gap:8px; justify-content:space-around;">
        <div>🏠<div style='font-size:12px;color:#2e7d32'>Home</div></div>
        <div>⚙️<div style='font-size:12px;color:#2e7d32'>Prefs</div></div>
        <div>🔍<div style='font-size:12px;color:#2e7d32'>Find</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Router
# -------------------------
screen = st.session_state.get("screen", "welcome")
if screen == "welcome":
    screen_welcome()
elif screen == "login":
    screen_login()
elif screen == "preferences":
    screen_preferences()
else:
    screen_recommend()

bottom_nav()
st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
