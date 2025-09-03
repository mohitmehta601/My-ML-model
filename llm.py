import json
import os
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv


def _get_gemini_client():
    # Lazy import to avoid dependency at training time
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "google-generativeai package is required. Install dependencies from requirements.txt"
        ) from e

    load_dotenv(override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it or add to a .env file."
        )

    # Configure the Gemini API
    genai.configure(api_key=api_key)
    return genai


def generate_recommendation_report(
    base_inputs: Dict[str, Any], predictions: Dict[str, Any], confidences: Dict[str, float]
) -> Dict[str, Any]:
    """
    Call the Gemini API to transform raw predictions into a user-facing, structured
    recommendation similar to the provided screenshot.
    Returns a dict (JSON-compatible) ready to render.
    """

    genai = _get_gemini_client()

    # Compute a representative confidence from Primary_Fertilizer if present
    primary_conf = confidences.get("Primary_Fertilizer", None)
    if primary_conf is not None:
        primary_conf_pct = round(float(primary_conf) * 100)
    else:
        primary_conf_pct = None

    sowing_date = base_inputs.get("Sowing_Date")
    field_size = base_inputs.get("Field_Size")
    field_unit = base_inputs.get("Field_Unit", "hectares")

    system_prompt = (
        "You are an agronomy assistant. Convert soil/crop + ML outputs into a clear,\n"
        "practical fertilizer recommendation. Return only JSON and keep values realistic.\n"
        "When estimating amounts and costs, scale by field size."
    )

    user_payload = {
        "inputs": base_inputs,
        "predictions": predictions,
        "primary_confidence_percent": primary_conf_pct,
        "format_requirements": {
            "type": "object",
            "properties": {
                "ml_model_prediction": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "confidence_percent": {"type": "number"},
                        "npk": {"type": "string"},
                    },
                },
                "soil_condition": {
                    "type": "object",
                    "properties": {
                        "ph_status": {"type": "string"},
                        "moisture_status": {"type": "string"},
                        "nutrient_deficiencies": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "primary_fertilizer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount_kg": {"type": "number"},
                        "reason": {"type": "string"},
                        "application_method": {"type": "string"},
                    },
                },
                "secondary_fertilizer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "amount_kg": {"type": "number"},
                        "reason": {"type": "string"},
                        "application_method": {"type": "string"},
                    },
                },
                "organic_alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "amount_kg": {"type": "number"},
                            "reason": {"type": "string"},
                            "timing": {"type": "string"},
                        },
                    },
                },
                "application_timing": {
                    "type": "object",
                    "properties": {
                        "primary": {"type": "string"},
                        "secondary": {"type": "string"},
                        "organics": {"type": "string"},
                    },
                },
                "cost_estimate": {
                    "type": "object",
                    "properties": {
                        "primary": {"type": "number"},
                        "secondary": {"type": "number"},
                        "organics": {"type": "number"},
                        "total": {"type": "number"},
                        "notes": {"type": "string"},
                    },
                },
            },
        },
        "notes": (
            "- Use predicted fertilizers/statuses as anchors.\n"
            "- If pH_Amendment is 'None', reflect that.\n"
            f"- Sowing date: {sowing_date}, field size: {field_size} {field_unit}.\n"
            "- Provide concise, farmer-friendly text."
        ),
    }

    full_prompt = (
        f"{system_prompt}\n\n"
        "Generate a structured agronomy report as JSON (no extra text).\n"
        + json.dumps(user_payload)
    )

    # Use Gemini Flash model (good balance of performance and cost)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Configure generation parameters
    generation_config = genai.types.GenerationConfig(
        temperature=0.4,
        candidate_count=1,
    )

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        content = response.text
        
        # Try to parse the JSON response
        try:
            data = json.loads(content)
        except Exception:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except Exception:
                    # Fallback: wrap content
                    data = {"raw": content}
            else:
                # Fallback: wrap content
                data = {"raw": content}

    except Exception as e:
        # Fallback response in case of API error
        data = {"error": f"Failed to generate recommendation: {str(e)}"}

    # Attach some context for rendering
    data["_meta"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "inputs": base_inputs,
        "predictions": predictions,
        "confidences": confidences,
    }
    return data

