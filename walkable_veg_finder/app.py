# app.py
from typing import List, Optional, Dict, Any, Tuple
from math import radians, sin, cos, asin, sqrt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import os
import googlemaps
from dotenv import load_dotenv
import threading
from datetime import datetime
import json
from fastapi import HTTPException
from collections import defaultdict

# ----------------------------
# Models
# ----------------------------
class MobilityPrefs(BaseModel):
    wheelchair: bool = False
    needs_ramp: bool = False
    needs_auto_door: bool = False
    needs_accessible_bathroom: bool = False


class FoodPrefs(BaseModel):
    vegetarian: bool = True
    vegan: bool = False
    cuisines: List[str] = Field(default_factory=list)
    disliked_ingredients: List[str] = Field(default_factory=list)
    budget_usd: Optional[float] = None


class OnboardingPayload(BaseModel):
    user_id: str
    food: FoodPrefs
    mobility: MobilityPrefs
    max_walk_miles: float = 2.0

    @field_validator("max_walk_miles")
    @classmethod
    def _validate_walk(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_walk_miles must be positive")
        return v


class Location(BaseModel):
    lat: float
    lon: float


class RecommendationRequest(BaseModel):
    user_id: str
    location: Location


class Dish(BaseModel):
    name: str
    price: Optional[float] = None
    description: str = ""


class Restaurant(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    cuisines: List[str]
    accessibility: Dict[str, bool]
    menu: List[Dish]


class MatchResult(BaseModel):
    restaurant_id: str
    restaurant_name: str
    distance_miles: float
    best_dish: Dish
    score: float


class RecommendationResponse(BaseModel):
    found: bool
    result: Optional[MatchResult] = None
    reason: Optional[str] = None


# ----------------------------
# App setup & in-memory storage
# ----------------------------
app = FastAPI(title="Walkable Veg Finder — API")

PREFERENCES: Dict[str, OnboardingPayload] = {}

MOCK_RESTAURANTS: List[Restaurant] = [
    Restaurant(
        id="r1",
        name="Green Slice Pizzeria",
        lat=40.7429,
        lon=-73.9925,
        cuisines=["italian", "pizza"],
        accessibility={"wheelchair": True, "ramp": True, "auto_door": False, "accessible_bathroom": True},
        menu=[
            Dish(name="Margherita Pizza", price=10.0, description="tomato, mozzarella, basil"),
            Dish(name="White Pie Ricotta", price=14.0, description="ricotta, mozzarella, garlic"),
            Dish(name="Pepperoni", price=13.0, description="pepperoni, mozzarella"),
        ],
    ),
    Restaurant(
        id="r2",
        name="Masala Ghar",
        lat=40.7441,
        lon=-73.9870,
        cuisines=["indian"],
        accessibility={"wheelchair": True, "ramp": True, "auto_door": True, "accessible_bathroom": True},
        menu=[
            Dish(name="Paneer Tikka", price=12.0, description="grilled paneer with spices"),
            Dish(name="Chana Masala", price=11.0, description="chickpeas in tomato gravy"),
            Dish(name="Butter Chicken", price=15.0, description="chicken in creamy gravy"),
        ],
    ),
    Restaurant(
        id="r3",
        name="Sushi Town",
        lat=40.7392,
        lon=-73.9901,
        cuisines=["japanese"],
        accessibility={"wheelchair": False, "ramp": False, "auto_door": False, "accessible_bathroom": False},
        menu=[
            Dish(name="Avocado Roll", price=8.0, description="avocado, rice, nori"),
            Dish(name="Salmon Nigiri", price=12.0, description="salmon over rice"),
        ],
    ),
]

# ----------------------------
# Utils: haversine
# ----------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = R_km * c
    return km * 0.621371


# --- ADD PHASE 3 FILTERING HERE ---
VEG_KEYWORDS = {
    "veg": ["vegetarian", "vegan", "paneer", "tofu", "veggie", "margherita", "chana", "chickpea", "dal", "lentil",
            "mushroom", "spinach", "avocado", "salad", "cheese", "ricotta", "mozzarella"],
    "nonveg": ["chicken", "beef", "pork", "fish", "salmon", "tuna", "pepperoni", "shrimp", "lamb"]
}

def is_veg_dish(text: str, vegetarian: bool, vegan: bool) -> bool:
    t = text.lower()
    if any(w in t for w in VEG_KEYWORDS["nonveg"]):
        return False
    if vegan:
        banned = ["cheese", "paneer", "butter", "ghee", "cream", "milk", "egg", "yogurt"]
        if any(b in t for b in banned):
            return False
        return True
    return vegetarian

def ingredient_blocklisted(text: str, block: List[str]) -> bool:
    t = text.lower()
    return any(b.lower() in t for b in block)

def accessibility_ok(restaurant: Restaurant, prefs: MobilityPrefs) -> bool:
    acc = restaurant.accessibility
    if prefs.wheelchair and not acc.get("wheelchair", False):
        return False
    if prefs.needs_ramp and not acc.get("ramp", False):
        return False
    if prefs.needs_auto_door and not acc.get("auto_door", False):
        return False
    if prefs.needs_accessible_bathroom and not acc.get("accessible_bathroom", False):
        return False
    return True

def cuisine_match_score(restaurant: Restaurant, desired: List[str]) -> float:
    if not desired:
        return 0.5
    desired_lower = {c.lower() for c in desired}
    rest_lower = {c.lower() for c in restaurant.cuisines}
    inter = desired_lower.intersection(rest_lower)
    return 1.0 if inter else 0.2

# ----------------------------
# Phase 4: Accessibility enrichment & reports
# ----------------------------
load_dotenv()
gmaps = None
try:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if key:
        gmaps = googlemaps.Client(key=key)
except Exception as e:
    print("Warning: googlemaps client not initialized:", e)
    gmaps = None

ACCESSIBILITY_REPORTS: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
PLACE_DETAILS_CACHE: Dict[str, Dict[str, Any]] = {}
PLACE_DETAILS_LOCK = threading.Lock()
REPORTS_PERSIST_FILE = "accessibility_reports.json"

# try to load persisted reports (optional)
def _load_reports_from_disk():
    try:
        if os.path.exists(REPORTS_PERSIST_FILE):
            with open(REPORTS_PERSIST_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                for k, v in data.items():
                    ACCESSIBILITY_REPORTS[k] = v
    except Exception as e:
        print("Could not load persisted reports:", e)

def _save_reports_to_disk():
    try:
        with open(REPORTS_PERSIST_FILE, "w", encoding="utf-8") as fh:
            json.dump(ACCESSIBILITY_REPORTS, fh, default=list, indent=2)
    except Exception as e:
        print("Could not persist reports:", e)

_load_reports_from_disk()

def merge_accessibility_from_google_and_reports(place: Dict[str, Any], restaurant_id: str) -> Tuple[Dict[str,bool], float]:
    acc = {
        "wheelchair": False,
        "ramp": False,
        "auto_door": False,
        "accessible_bathroom": False
    }

    # 1) Structured google fields & text scan
    if place:
        details = place.get("result", {}) if isinstance(place, dict) else {}
        if details.get("wheelchair_accessible_entrance") is True:
            acc["wheelchair"] = True

        description_text = " ".join([
            str(details.get("name", "")),
            str(details.get("formatted_address", "")),
            str(details.get("vicinity", "") or ""),
            str(details.get("editorial_summary", {}).get("overview", "") if isinstance(details.get("editorial_summary"), dict) else "")
        ]).lower()

        for review in details.get("reviews", [])[:5]:
            description_text += " " + (review.get("text", "") or "")

        if "wheelchair" in description_text or "accessible" in description_text or "accessibility" in description_text:
            acc["wheelchair"] = True

        if "ramp" in description_text:
            acc["ramp"] = True
        if "automatic door" in description_text or "auto door" in description_text or "automatic entrance" in description_text:
            acc["auto_door"] = True
        if "accessible bathroom" in description_text or "accessible restroom" in description_text or "disabled toilet" in description_text:
            acc["accessible_bathroom"] = True

    # 2) Crowd reports
    reports = ACCESSIBILITY_REPORTS.get(restaurant_id, [])
    positives = 0
    negatives = 0
    total = 0
    for r in reports:
        total += 1
        for k in acc.keys():
            if k in r and r[k] is not None:
                if r[k]:
                    positives += 1
                else:
                    negatives += 1

    # 3) Heuristic score calculation
    base = 0.0
    if acc["wheelchair"]:
        base += 0.6
    if acc["ramp"]:
        base += 0.1
    if acc["auto_door"]:
        base += 0.15
    if acc["accessible_bathroom"]:
        base += 0.15

    base_score = min(1.0, base)

    if total > 0:
        crowd_score = max(-1.0, min(1.0, (positives - negatives) / total))
        crowd_score = (crowd_score + 1.0) / 2.0
        final = 0.6 * base_score + 0.4 * crowd_score
    else:
        final = base_score

    final = max(0.0, min(1.0, final))
    return acc, round(final, 3)

@app.post("/report_accessibility")
def report_accessibility(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload:
    {
      "restaurant_id": "r1",
      "wheelchair": true/false (optional),
      "ramp": true/false (optional),
      "auto_door": true/false (optional),
      "accessible_bathroom": true/false (optional),
      "notes": "...",
      "reporter_id": "user123"  (optional)
    }
    """
    rid = report.get("restaurant_id")
    if not rid:
        raise HTTPException(status_code=400, detail="restaurant_id is required")

    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "wheelchair": bool(report.get("wheelchair")) if "wheelchair" in report else None,
        "ramp": bool(report.get("ramp")) if "ramp" in report else None,
        "auto_door": bool(report.get("auto_door")) if "auto_door" in report else None,
        "accessible_bathroom": bool(report.get("accessible_bathroom")) if "accessible_bathroom" in report else None,
        "notes": report.get("notes"),
        "reporter_id": report.get("reporter_id")
    }

def fetch_place_details_safe(place_id: str) -> Dict[str, Any]:
    if not gmaps:
        return {}
    with PLACE_DETAILS_LOCK:
        if place_id in PLACE_DETAILS_CACHE:
            return PLACE_DETAILS_CACHE[place_id]
    try:
        details = gmaps.place(place_id=place_id, fields=["name", "formatted_address", "reviews", "wheelchair_accessible_entrance", "vicinity", "editorial_summary"])
    except Exception as e:
        print(f"Places details error for {place_id}: {e}")
        details = {}
    with PLACE_DETAILS_LOCK:
        try:
            PLACE_DETAILS_CACHE[place_id] = details
        except Exception:
            pass
    return details

def fetch_nearby_restaurants_with_details(lat: float, lon: float, radius_meters: int = 1500) -> List[Restaurant]:
    raw_places = []
    if gmaps:
        try:
            res = gmaps.places_nearby(
                location=(lat, lon),
                radius=radius_meters,
                type="restaurant",
                keyword="vegetarian"
            )
            raw_places = res.get("results", [])
        except Exception as e:
            print(f"Google API error (nearby): {e}")
            raw_places = []
    else:
        raw_places = []

    restaurants: List[Restaurant] = []
    for place in raw_places:
        place_id = place.get("place_id")
        details = fetch_place_details_safe(place_id) if place_id else {}
        merged_acc, acc_score = merge_accessibility_from_google_and_reports(details, place_id or place.get("name", "unknown"))
        types = place.get("types", []) or []
        cuisines = [t.replace("_", " ") for t in types if t not in ("restaurant", "food", "point_of_interest")][:3]

        restaurants.append(Restaurant(
            id=place_id or place.get("name", ""),
            name=place.get("name", "Unknown"),
            lat=place["geometry"]["location"]["lat"],
            lon=place["geometry"]["location"]["lng"],
            cuisines=cuisines,
            accessibility=merged_acc,
            menu=[Dish(name="Vegetarian Option", description="Placeholder from Google API")],
        ))

    return restaurants

# ----------------------------
# Ranking (updated to include accessibility)
# ----------------------------
def rank_score(distance_miles: float, cuisine_score: float, dish_price: Optional[float], budget: Optional[float],
               accessibility_score: Optional[float] = None) -> float:
    dist_component = max(0.0, 1.2 - (distance_miles / 2.0))
    cuisine_component = cuisine_score
    price_component = 0.5
    if budget is not None and dish_price is not None:
        price_component = 1.0 if dish_price <= budget else max(0.1, 1.0 - (dish_price - budget) / max(2*budget, 10))
    if accessibility_score is None:
        accessibility_score = 0.5
    final = (0.45 * dist_component) + (0.25 * cuisine_component) + (0.15 * price_component) + (0.15 * accessibility_score)
    return round(final, 4)

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/onboarding")
def save_onboarding(prefs: OnboardingPayload) -> Dict[str, Any]:
    PREFERENCES[prefs.user_id] = prefs
    return {"ok": True, "message": "Preferences saved."}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest) -> RecommendationResponse:
    prefs = PREFERENCES.get(req.user_id)
    if not prefs:
        raise HTTPException(status_code=400, detail="User preferences not found. Call /onboarding first.")

    # Phase 2/4: get real restaurants (with details) if possible
    real_restaurants = fetch_nearby_restaurants_with_details(req.location.lat, req.location.lon)

    candidates: List[MatchResult] = []

    # Check both Google API + mock data (so it still works without API calls)
    for r in (real_restaurants + MOCK_RESTAURANTS):
        # compute distance
        dist = haversine_miles(req.location.lat, req.location.lon, r.lat, r.lon)
        if dist > prefs.max_walk_miles:
            continue
        if not accessibility_ok(r, prefs.mobility):
            continue

        # choose best dish by simple scan & filters
        best_dish: Optional[Dish] = None
        for d in r.menu:
            text = f"{d.name} {d.description}"
            if not is_veg_dish(text, prefs.food.vegetarian, prefs.food.vegan):
                continue
            if ingredient_blocklisted(text, prefs.food.disliked_ingredients):
                continue
            if prefs.food.budget_usd and d.price and d.price > prefs.food.budget_usd * 1.5:
                continue
            best_dish = d
            break

        # fallback: if Google placeholders exist, accept them (they may be veg)
        if not best_dish and r.menu:
            # try first item if veg/allowed
            d = r.menu[0]
            text = f"{d.name} {d.description}"
            if is_veg_dish(text, prefs.food.vegetarian, prefs.food.vegan) and not ingredient_blocklisted(text, prefs.food.disliked_ingredients):
                best_dish = d

        if not best_dish:
            continue

        cscore = cuisine_match_score(r, prefs.food.cuisines)

        # obtain accessibility score (best-effort) — if we have reports or cached place details, use merge helper
        acc_score = None
        if r.id and (r.id in ACCESSIBILITY_REPORTS or r.id in PLACE_DETAILS_CACHE):
            details = PLACE_DETAILS_CACHE.get(r.id, {})
            _, acc_score = merge_accessibility_from_google_and_reports(details, r.id)
        else:
            # neutral if unknown
            acc_score = 0.5

        s = rank_score(dist, cscore, best_dish.price, prefs.food.budget_usd, accessibility_score=acc_score)

        candidates.append(MatchResult(
            restaurant_id=r.id,
            restaurant_name=r.name,
            distance_miles=round(dist, 2),
            best_dish=best_dish,
            score=s,
        ))

    if not candidates:
        return RecommendationResponse(found=False, reason="No walkable, accessible vegetarian matches found.")

    candidates.sort(key=lambda x: (-x.score, x.distance_miles))
    return RecommendationResponse(found=True, result=candidates[0])

@app.get("/")
def root():
    return {"service": "Walkable Veg Finder API", "status": "ok"}

POINTS_STORE_FILE = "points.json"
POINTS_LOCK = threading.Lock()

# in-memory user points: {user_id: {"points": int, "last_report": iso, "streak": int}}
_USER_POINTS = {}

def _load_points_from_disk():
    try:
        if os.path.exists(POINTS_STORE_FILE):
            with open(POINTS_STORE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                for k, v in data.items():
                    _USER_POINTS[k] = v
    except Exception as e:
        print("Could not load points file:", e)

def _save_points_to_disk():
    try:
        with POINTS_LOCK:
            with open(POINTS_STORE_FILE, "w", encoding="utf-8") as fh:
                json.dump(_USER_POINTS, fh, indent=2)
    except Exception as e:
        print("Could not persist points:", e)

# initialize (safe if file missing)
_load_points_from_disk()

# badge thresholds (pts -> badge name)
BADGE_THRESHOLDS = [
    (100, "🏆 Accessibility Hero"),
    (50,  "🏅 Accessibility Champion"),
    (20,  "🌱 Eco Explorer"),
]

def award_points(user_id: str, pts: int, reason: str = "") -> int:
    """Award points to a user and persist asynchronously. Returns new total points."""
    with POINTS_LOCK:
        u = _USER_POINTS.get(user_id, {"points": 0, "last_report": None, "streak": 0})
        # streak logic: +1 if last_report was yesterday, reset if gap >1 day
        try:
            if u.get("last_report"):
                last_date = datetime.fromisoformat(u["last_report"]).date()
                delta = (datetime.utcnow().date() - last_date).days
                if delta == 1:
                    u["streak"] = u.get("streak", 0) + 1
                elif delta == 0:
                    pass
                else:
                    u["streak"] = 1
            else:
                u["streak"] = 1
        except Exception:
            u["streak"] = u.get("streak", 0) + 1

        u["points"] = u.get("points", 0) + int(pts)
        u["last_report"] = datetime.utcnow().isoformat()
        _USER_POINTS[user_id] = u

    # persist in background
    try:
        threading.Thread(target=_save_points_to_disk, daemon=True).start()
    except Exception:
        pass

    return _USER_POINTS[user_id]["points"]

def get_user_points(user_id: str) -> dict:
    return _USER_POINTS.get(user_id, {"points": 0, "last_report": None, "streak": 0})

def get_badges_for_user(user_id: str) -> list:
    pts = get_user_points(user_id).get("points", 0)
    badges = [name for thr, name in BADGE_THRESHOLDS if pts >= thr]
    # order small->large threshold
    badges_sorted = sorted(badges, key=lambda n: next((t for t, nm in BADGE_THRESHOLDS if nm == n), 0))
    return badges_sorted

def get_leaderboard(top_n: int = 10) -> list:
    items = [{"user_id": uid, "points": info.get("points", 0), "streak": info.get("streak", 0)}
             for uid, info in _USER_POINTS.items()]
    items.sort(key=lambda x: (-x["points"], -x["streak"]))
    return items[:top_n]

# server-side eco-score model (simple heuristic)
def compute_eco_score_server(distance_miles: float, dish_name: str = "") -> dict:
    dist_component = max(0.0, 1.2 - (distance_miles / 2.0))  # closer -> higher
    dish = (dish_name or "").lower()
    plant_keywords = ("paneer","chana","tofu","salad","avocado","vegetarian","veggie","lentil","dal","beans","chickpea")
    dairy_keywords = ("cheese","butter","cream","milk","ghee","paneer")
    plant_bonus = 0.2 if any(k in dish for k in plant_keywords) else 0.0
    dairy_penalty = -0.05 if any(k in dish for k in dairy_keywords) else 0.0
    raw = dist_component + plant_bonus + dairy_penalty
    eco = max(0.0, min(1.0, raw))
    return {
        "eco_score": round(eco, 3),
        "components": {
            "distance_component": round(dist_component, 3),
            "plant_bonus": round(plant_bonus, 3),
            "dairy_penalty": round(dairy_penalty, 3)
        }
    }

# ----------------------------
# New endpoints: points, badges, leaderboard, eco_score
# ---------------------
POINTS_STORE_FILE = "points.json"
POINTS_LOCK = threading.Lock()

# in-memory user points: {user_id: {"points": int, "last_report": iso, "streak": int}}
_USER_POINTS = {}

def _load_points_from_disk():
    try:
        if os.path.exists(POINTS_STORE_FILE):
            with open(POINTS_STORE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                for k, v in data.items():
                    _USER_POINTS[k] = v
    except Exception as e:
        print("Could not load points file:", e)

def _save_points_to_disk():
    try:
        with POINTS_LOCK:
            with open(POINTS_STORE_FILE, "w", encoding="utf-8") as fh:
                json.dump(_USER_POINTS, fh, indent=2)
    except Exception as e:
        print("Could not persist points:", e)

# initialize (safe if file missing)
_load_points_from_disk()

# badge thresholds (pts -> badge name)
BADGE_THRESHOLDS = [
    (100, "🏆 Accessibility Hero"),
    (50,  "🏅 Accessibility Champion"),
    (20,  "🌱 Eco Explorer"),
]

def award_points(user_id: str, pts: int, reason: str = "") -> int:
    """Award points to a user and persist asynchronously. Returns new total points."""
    with POINTS_LOCK:
        u = _USER_POINTS.get(user_id, {"points": 0, "last_report": None, "streak": 0})
        # streak logic: +1 if last_report was yesterday, reset if gap >1 day
        try:
            if u.get("last_report"):
                last_date = datetime.fromisoformat(u["last_report"]).date()
                delta = (datetime.utcnow().date() - last_date).days
                if delta == 1:
                    u["streak"] = u.get("streak", 0) + 1
                elif delta == 0:
                    pass
                else:
                    u["streak"] = 1
            else:
                u["streak"] = 1
        except Exception:
            u["streak"] = u.get("streak", 0) + 1

        u["points"] = u.get("points", 0) + int(pts)
        u["last_report"] = datetime.utcnow().isoformat()
        _USER_POINTS[user_id] = u

    # persist in background
    try:
        threading.Thread(target=_save_points_to_disk, daemon=True).start()
    except Exception:
        pass

    return _USER_POINTS[user_id]["points"]

def get_user_points(user_id: str) -> dict:
    return _USER_POINTS.get(user_id, {"points": 0, "last_report": None, "streak": 0})

def get_badges_for_user(user_id: str) -> list:
    pts = get_user_points(user_id).get("points", 0)
    badges = [name for thr, name in BADGE_THRESHOLDS if pts >= thr]
    # order small->large threshold
    badges_sorted = sorted(badges, key=lambda n: next((t for t, nm in BADGE_THRESHOLDS if nm == n), 0))
    return badges_sorted

def get_leaderboard(top_n: int = 10) -> list:
    items = [{"user_id": uid, "points": info.get("points", 0), "streak": info.get("streak", 0)}
             for uid, info in _USER_POINTS.items()]
    items.sort(key=lambda x: (-x["points"], -x["streak"]))
    return items[:top_n]

# server-side eco-score model (simple heuristic)
def compute_eco_score_server(distance_miles: float, dish_name: str = "") -> dict:
    dist_component = max(0.0, 1.2 - (distance_miles / 2.0))  # closer -> higher
    dish = (dish_name or "").lower()
    plant_keywords = ("paneer","chana","tofu","salad","avocado","vegetarian","veggie","lentil","dal","beans","chickpea")
    dairy_keywords = ("cheese","butter","cream","milk","ghee","paneer")
    plant_bonus = 0.2 if any(k in dish for k in plant_keywords) else 0.0
    dairy_penalty = -0.05 if any(k in dish for k in dairy_keywords) else 0.0
    raw = dist_component + plant_bonus + dairy_penalty
    eco = max(0.0, min(1.0, raw))
    return {
        "eco_score": round(eco, 3),
        "components": {
            "distance_component": round(dist_component, 3),
            "plant_bonus": round(plant_bonus, 3),
            "dairy_penalty": round(dairy_penalty, 3)
        }
    }

# ----------------------------
# New endpoints: points, badges, leaderboard, eco_score
# ---------------------