import json
import pytest
from unittest.mock import patch, MagicMock
from app import app, get_command_type, extract_route, resolve_place, get_directions_steps


# -------------------- TEST FIXTURE --------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


# -------------------- UNIT TESTS --------------------


# --- Command Parsing ---

def test_get_command_type_valid_variants():
    assert get_command_type("walk from A to B") == "WALK"
    assert get_command_type("   DRIVE from X to Y") == "DRIVE"
    assert get_command_type("Transit from 1st St to Union Sq") == "TRANSIT"
    assert get_command_type("help") == "HELP"
    assert get_command_type("   help please") == "HELP"

def test_get_command_type_unknown():
    assert get_command_type("fly to Mars") == "UNKNOWN"
    assert get_command_type("") == "UNKNOWN"
    assert get_command_type("running from dogs") == "UNKNOWN"


# --- GPT Route Extraction ---

@patch("app.client.chat.completions.create")
def test_extract_route_valid(mock_gpt):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "origin": "Statue of Liberty",
            "destination": "Empire State Building"
        })))
    ]
    mock_gpt.return_value = mock_response

    result = extract_route("walk from statue of liberty to empire state building")
    assert result["origin"] == "Statue of Liberty"
    assert result["destination"] == "Empire State Building"

@patch("app.client.chat.completions.create")
def test_extract_route_malformed_json(mock_gpt):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="this is not json"))
    ]
    mock_gpt.return_value = mock_response

    with pytest.raises(ValueError, match="LLM response could not be parsed as JSON"):
        extract_route("walk from x to y")


# --- Google Places ---

def test_resolve_place_valid(requests_mock):
    requests_mock.post(
        "https://places.googleapis.com/v1/places:searchText",
        json={
            "places": [{
                "location": {"latitude": 40.7128, "longitude": -74.0060}
            }]
        }
    )
    result = resolve_place("New York")
    assert result == {"latitude": 40.7128, "longitude": -74.0060}

def test_resolve_place_no_results(requests_mock):
    requests_mock.post(
        "https://places.googleapis.com/v1/places:searchText",
        json={"places": []}
    )
    with pytest.raises(ValueError, match="Could not resolve place:"):
        resolve_place("Nowhereville")

def test_resolve_place_missing_key(requests_mock):
    requests_mock.post(
        "https://places.googleapis.com/v1/places:searchText",
        json={}
    )
    with pytest.raises(ValueError, match="Could not resolve place:"):
        resolve_place("EmptyResponseLand")


# --- Google Directions ---

def test_get_directions_steps_valid(requests_mock):
    requests_mock.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        json={
            "routes": [{
                "legs": [{
                    "steps": [
                        {
                            "html_instructions": "Turn <b>left</b> on Main St",
                            "distance": {"text": "0.5 mi"}
                        },
                        {
                            "html_instructions": "Go <b>straight</b> 2 blocks",
                            "distance": {"text": "0.3 mi"}
                        }
                    ]
                }]
            }]
        }
    )
    output = get_directions_steps("A", "B", "walk")
