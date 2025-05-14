import os
import re
import json
import requests
from flask import Flask, request, Response, render_template
from dotenv import load_dotenv
from openai import OpenAI
from xml.sax.saxutils import escape


# ------------ APP AND CLIENT ------------

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


# ------------ PARSE COMMAND ------------

def get_command_type(message: str) -> tuple[str, str]:
    """Returns (internal label, Google API mode)."""
    msg = message.strip().lower()
    if msg.startswith("help"):
        return ("HELP", "")
    if msg.startswith("walk"):
        return ("WALK", "walking")
    if msg.startswith("transit"):
        return ("TRANSIT", "transit")
    if msg.startswith("drive"):
        return ("DRIVE", "driving")
    return ("UNKNOWN", "")


# ------------ EXTRACT PLACES ------------

def extract_route(message: str) -> dict:
    """Uses GPT to extract and normalize the origin and destination from free-form text."""
    prompt = f"""
    A user sent: '{message}'
    Extract just the origin and destination. Fix any misspellings or informal place names.
    Respond only in JSON:
    {{
      "origin": "place name suitable for Google search",
      "destination": "place name suitable for Google search"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        raise ValueError("LLM response could not be parsed as JSON.")


def resolve_place(query: str, lat=None, lng=None) -> dict:
    """Resolves a place to a walkable lat/lng using Google Places API with bias."""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.location,places.id"
    }

    body = {"textQuery": query}
    if lat is not None and lng is not None:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 5000  # tighter bias for walking
            }
        }

    response = requests.post(url, headers=headers, json=body, timeout=5)
    data = response.json()

    if "places" not in data or not data["places"]:
        raise ValueError(f"Could not resolve place: {query}")

    place = data["places"][0]
    return {
        "name": place["displayName"]["text"],
        "lat": place["location"]["latitude"],
        "lng": place["location"]["longitude"]
    }


# ------------ GET DIRECTIONS ------------

def get_directions_steps(origin_name: str, destination_name: str, mode: str) -> tuple[str, str]:
    """Get directions between origin and destination using Directions API"""
    origin = resolve_place(origin_name)
    destination = resolve_place(destination_name, lat=origin["lat"], lng=origin["lng"])

    origin_str = f"{origin['lat']},{origin['lng']}"
    destination_str = f"{destination['lat']},{destination['lng']}"

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin_str,
        "destination": destination_str,
        "mode": mode,
        "departure_time": "now",
        "key": GOOGLE_MAPS_API_KEY
    }

    response = requests.get(url, params=params, timeout=5)
    data = response.json()

    if not data.get("routes"):
        return "No route found."

    duration = data["routes"][0]["legs"][0]["duration"]["text"]
    steps = data["routes"][0]["legs"][0]["steps"]
    instructions = []
    for idx, step in enumerate(steps, start=1):
        text = re.sub(r"<.*?>", "", step.get("html_instructions", ""))
        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
        dist = step["distance"]["text"]
        instructions.append(f"{idx}. {text} ({dist})")

    return duration, "\n".join(instructions)


# ------------ CHUNK SMS ------------

def split_sms(text: str, max_len: int = 1600) -> list[str]:
    """Splits text into chunks that fit in individual SMS messages."""
    parts = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    parts.append(text)
    return parts


# ------------ API ROUTES ------------

@app.route("/", methods=["GET"])
def index():
    """Render landing page."""
    return render_template("index.html")

@app.route("/privacy")
def privacy():
    """Render privacy policy."""
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    """Render terms of service."""
    return render_template("terms.html")

@app.route("/sms", methods=["POST"])
def handle_sms():
    """Main route to handle incoming SMS and return directions or help text."""
    user_input = request.form.get("Body", "").strip()
    command, mode = get_command_type(user_input)

    if command == "HELP":
        message = (
            "SMS Directions:\n"
            "Commands:\n"
            "1. 'WALK from [A] to [B]'\n"
            "2. 'TRANSIT from [A] to [B]'\n"
            "3. 'DRIVE from [A] to [B]'\n"
            "4. For more info, visit sms-directions-production.up.railway.app."
        )
    elif command in {"WALK", "TRANSIT", "DRIVE"}:
        try:
            route = extract_route(user_input)
            duration, steps = get_directions_steps(route["origin"], route["destination"], mode)
            message = (
                "SMS Directions:\n"
                f"From: {route['origin']}\n"
                f"To: {route['destination']}\n"
                f"Mode: {command}\n"
                f"Duration: {duration}\n\n"
                f"{steps}"
            )
        except (ValueError, KeyError) as e:
            message = f"Error: {str(e)}"
    else:
        message = "SMS Directions:\nUnrecognized command. Type 'HELP' for instructions."

    chunks = split_sms(message)
    xml_messages = "".join(f"<Message>{escape(chunk)}</Message>" for chunk in chunks)
    xml_response = f"<Response>{xml_messages}</Response>"

    return Response(xml_response, mimetype="application/xml")


# ------------ RUN APP ------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)




#### TO DO
# TWILIO CAMPAIGN,
# TRANSIT BUS/TRAIN NUMBERS 
# QUALITY OF LIFE CHANGES TO MINIMIZE DIRECTIONS
# &amp; AND SIMILAR CHARACTERS