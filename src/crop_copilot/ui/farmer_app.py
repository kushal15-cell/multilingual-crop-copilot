from __future__ import annotations

import hashlib
import os

import httpx
import streamlit as st

from crop_copilot.ui.locations import KARNATAKA_LOCATIONS, resolve_location

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
LANGUAGES = {"English": "en", "ಕನ್ನಡ": "kn", "हिन्दी": "hi"}

st.set_page_config(page_title="Crop Health Copilot", page_icon="🌿")
st.title("🌿 Crop Health Copilot")
st.caption("Preliminary crop-health support with agronomist review for risky advice")

language_name = st.selectbox("Language / ಭಾಷೆ / भाषा", list(LANGUAGES))
language = LANGUAGES[language_name]
crop = st.text_input("Crop", value="tomato")
location_name = st.selectbox(
    "District / nearest market",
    list(KARNATAKA_LOCATIONS),
    index=list(KARNATAKA_LOCATIONS).index("Davangere"),
)
selected_location = KARNATAKA_LOCATIONS[location_name]
with st.expander("Optional: use precise coordinates"):
    latitude = st.number_input(
        "Latitude",
        value=float(selected_location["latitude"]),
        format="%.6f",
        key=f"latitude_{location_name}",
    )
    longitude = st.number_input(
        "Longitude",
        value=float(selected_location["longitude"]),
        format="%.6f",
        key=f"longitude_{location_name}",
    )
location = resolve_location(location_name, latitude, longitude)
st.caption(
    "Weather uses the selected coordinates. Market prices in this prototype are simulated "
    "and are not live mandi quotes."
)
image = st.file_uploader("Photograph of the affected leaf", type=["jpg", "jpeg", "png", "webp"])
if image is not None:
    st.image(image, caption="Submitted image", width=360)

audio = st.audio_input("Ask by voice (optional)")
if audio is not None:
    audio_digest = hashlib.sha256(audio.getvalue()).hexdigest()
else:
    audio_digest = None
if audio is not None and audio_digest != st.session_state.get("last_audio_digest"):
    st.session_state["last_audio_digest"] = audio_digest
    with st.spinner("Transcribing..."):
        response = httpx.post(
            f"{API_BASE}/v1/speech/transcribe",
            files={"audio": ("question.wav", audio.getvalue(), audio.type)},
            data={"language": language},
            timeout=300,
        )
    if response.is_success:
        st.session_state["question"] = response.json()["text"]
        st.rerun()
    else:
        st.error(response.json().get("detail", response.text))

question = st.text_area(
    "Question",
    value=st.session_state.get("question", "What should I do next?"),
    height=100,
)

if st.button("Check crop", type="primary", disabled=image is None):
    with st.spinner("Checking the image and context..."):
        response = httpx.post(
            f"{API_BASE}/v1/advice",
            files={"image": (image.name, image.getvalue(), image.type)},
            data={
                "crop": crop,
                "question": question,
                "language": language,
                "latitude": str(location["latitude"]),
                "longitude": str(location["longitude"]),
                "region": location["name"],
                "market": location["market"],
            },
            timeout=120,
        )
    if not response.is_success:
        st.error(response.json().get("detail", response.text))
    else:
        result = response.json()
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    status = result["status"]
    st.subheader("Result")
    st.write(result["farmer_message"])
    metadata = result.get("metadata", {})
    if metadata.get("market_is_synthetic"):
        st.warning("SIMULATED MARKET DATA — NOT LIVE MANDI PRICES.")
        with st.expander("Demonstration market context"):
            st.write(f"Market: {metadata.get('market_name', 'unavailable')}")
            st.write(f"Trend: {metadata.get('market_trend', 'unavailable')}")
            st.write(
                "Latest simulated modal price: "
                f"₹{metadata.get('market_latest_modal_price', 'unavailable')} per quintal"
            )
            st.write(
                "Simulated 7-day model forecast: "
                f"₹{metadata.get('market_predicted_7d_price', 'unavailable')} per quintal"
            )
    prediction = result.get("prediction")
    if prediction:
        st.metric("Image-model confidence", f"{prediction['confidence']:.0%}")
        if status == "need_more_information":
            st.caption("No disease label is released below the safety threshold.")
        else:
            st.caption(f"Preliminary label: {prediction['display_name']}")
        if prediction.get("model_version") == "demo-not-a-diagnosis":
            st.error(
                "DEMO MODE: this label is fixed test data and is not a diagnosis of the "
                "uploaded image. Train the PlantDoc model before real use."
            )
    if status == "pending_review":
        st.warning("This response is waiting for agronomist review.")
        review_id = result["review_id"]
        st.code(review_id, language=None)
        if st.button("Refresh review status"):
            refreshed = httpx.get(f"{API_BASE}/v1/advice/reviews/{review_id}", timeout=30)
            if refreshed.is_success:
                st.session_state["last_result"] = refreshed.json()
                st.rerun()
            else:
                st.error(refreshed.json().get("detail", refreshed.text))
    if status in {"ready", "approved"} and st.button("Play response"):
        audio_response = httpx.post(
            f"{API_BASE}/v1/speech/synthesize",
            params={"text": result["farmer_message"], "language": language},
            timeout=120,
        )
        if audio_response.is_success:
            st.audio(audio_response.content, format="audio/mp3")
        else:
            st.error(audio_response.json().get("detail", audio_response.text))
    if status in {"ready", "approved", "need_more_information"}:
        helpful_col, unhelpful_col = st.columns(2)
        with helpful_col:
            if st.button("Helpful", key=f"helpful_{result['request_id']}"):
                httpx.post(
                    f"{API_BASE}/v1/feedback",
                    json={"request_id": result["request_id"], "helpful": True},
                    timeout=15,
                )
                st.success("Feedback recorded")
        with unhelpful_col:
            if st.button("Not helpful", key=f"unhelpful_{result['request_id']}"):
                httpx.post(
                    f"{API_BASE}/v1/feedback",
                    json={
                        "request_id": result["request_id"],
                        "helpful": False,
                        "reason": "other",
                    },
                    timeout=15,
                )
                st.success("Feedback recorded")

st.divider()
st.caption(
    "Do not use this application for emergency poisoning, legal pesticide compliance, or a "
    "confirmed diagnosis. Contact a qualified local agricultural professional."
)
