from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Crop Health Copilot", page_icon="🌿", layout="centered")

# Cloud-safe defaults must be set before the application dependencies are imported.
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("ALLOW_DEMO_CLASSIFIER", "false")
os.environ.setdefault("CV_CHECKPOINT", "deploy_assets/best_model.pt")
os.environ.setdefault("CV_CLASS_NAMES", "deploy_assets/class_names.json")
os.environ.setdefault("PRICE_MODEL_PATH", "deploy_assets/price_forecaster.joblib")
os.environ.setdefault("MARKET_DATA_PATH", "deploy_assets/market_prices.csv")
os.environ.setdefault("DATABASE_URL", "sqlite:///./deploy_assets/review_queue.sqlite3")
os.environ.setdefault("EVENTS_PATH", "deploy_assets/events.jsonl")
os.environ.setdefault("WHISPER_MODEL_SIZE", "base")

from crop_copilot.api.dependencies import (  # noqa: E402
    get_event_logger,
    get_orchestrator,
    get_review_store,
    get_stt_service,
)
from crop_copilot.schemas import (  # noqa: E402
    AdviceDraft,
    AdviceRequest,
    AdviceStatus,
    Location,
)
from crop_copilot.services.speech import GTTSService  # noqa: E402
from crop_copilot.ui.locations import (  # noqa: E402
    KARNATAKA_LOCATIONS,
    resolve_location,
)

LANGUAGES = {"English": "en", "ಕನ್ನಡ": "kn", "हिन्दी": "hi"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEMO_CASES_PATH = Path("deploy_assets/demo_cases/manifest.json")


@st.cache_resource(show_spinner="Loading the calibrated PlantDoc model...")
def hosted_services():
    return get_orchestrator(), get_review_store(), get_event_logger()


def reviewer_password() -> str | None:
    try:
        return str(st.secrets["REVIEWER_PASSWORD"])
    except (FileNotFoundError, KeyError):
        return os.getenv("REVIEWER_PASSWORD")


def save_upload(upload) -> Path:
    suffix = Path(upload.name).suffix.casefold()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Upload a JPG, PNG, or WebP image")
    payload = upload.getvalue()
    if not payload:
        raise ValueError("The uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded image exceeds 10 MB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(payload)
        return Path(handle.name)


def save_audio(upload) -> Path:
    payload = upload.getvalue()
    if not payload:
        raise ValueError("The recorded audio is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The recorded audio exceeds 10 MB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        handle.write(payload)
        return Path(handle.name)


def load_demo_cases() -> list[dict[str, object]]:
    if not DEMO_CASES_PATH.exists():
        return []
    payload = json.loads(DEMO_CASES_PATH.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def render_farmer() -> None:
    st.subheader("Farmer advisory")
    st.caption("Preliminary decision support with calibrated abstention and human review")

    language_name = st.selectbox("Language / ಭಾಷೆ / भाषा", list(LANGUAGES))
    language = LANGUAGES[language_name]
    crop = st.selectbox("Crop", ["tomato"], help="The current model supports tomato only.")
    location_name = st.selectbox(
        "District / nearest market",
        list(KARNATAKA_LOCATIONS),
        index=list(KARNATAKA_LOCATIONS).index("Davangere"),
    )
    selected_location = KARNATAKA_LOCATIONS[location_name]
    with st.expander("Optional: use precise coordinates"):
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(selected_location["latitude"]),
            format="%.6f",
            key=f"hosted_latitude_{location_name}",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(selected_location["longitude"]),
            format="%.6f",
            key=f"hosted_longitude_{location_name}",
        )
    location = resolve_location(location_name, latitude, longitude)
    st.caption(
        "Weather uses the selected coordinates. Market prices are simulated demo data, not "
        "live mandi quotes."
    )

    demo_cases = load_demo_cases()
    input_modes = ["Upload my image"]
    if demo_cases:
        input_modes.append("Held-out PlantDoc demo")
    input_mode = st.radio("Image source", input_modes, horizontal=True)
    image = None
    demo_path: Path | None = None
    selected_demo: dict[str, object] | None = None
    if input_mode == "Held-out PlantDoc demo":
        case_labels = [str(case["title"]) for case in demo_cases]
        case_title = st.selectbox("Demo case", case_labels)
        selected_demo = next(case for case in demo_cases if case["title"] == case_title)
        demo_path = Path("deploy_assets/demo_cases") / str(selected_demo["filename"])
        st.image(str(demo_path), caption="Held-out official PlantDoc test image", width=360)
        st.info(
            "This image was not used for training. Its expected behavior was measured before "
            "deployment; it is included only for a reproducible walkthrough."
        )
    else:
        image = st.file_uploader(
            "Clear photograph of an affected leaf",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use daylight and make the affected leaf fill most of the frame.",
        )
        if image is not None:
            st.image(image, caption="Submitted image", width=360)
    audio = st.audio_input("Ask by voice (optional)")
    audio_digest = hashlib.sha256(audio.getvalue()).hexdigest() if audio is not None else None
    if audio is not None and audio_digest != st.session_state.get("hosted_audio_digest"):
        st.session_state["hosted_audio_digest"] = audio_digest
        audio_path: Path | None = None
        try:
            audio_path = save_audio(audio)
            with st.spinner("Transcribing locally; the first request downloads the model..."):
                st.session_state["hosted_question"] = get_stt_service().transcribe(
                    audio_path, language
                )
            st.rerun()
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
    if "hosted_question" not in st.session_state:
        st.session_state["hosted_question"] = "What should I do next?"
    if selected_demo is not None:
        demo_key = str(selected_demo["id"])
        if demo_key != st.session_state.get("selected_demo_key"):
            st.session_state["selected_demo_key"] = demo_key
            st.session_state["hosted_question"] = str(selected_demo["recommended_question"])
    question = st.text_area("Question", height=100, key="hosted_question")

    image_available = image is not None or demo_path is not None
    if st.button("Check crop", type="primary", disabled=not image_available):
        st.session_state.pop("hosted_result", None)
        image_path: Path | None = None
        temporary_upload = False
        try:
            if demo_path is not None:
                image_path = demo_path
            else:
                image_path = save_upload(image)
                temporary_upload = True
            request = AdviceRequest(
                crop=crop,
                question=question,
                language=language,
                location=Location(
                    latitude=location["latitude"],
                    longitude=location["longitude"],
                    region=location["name"],
                    market=location["market"],
                ),
                image_path=str(image_path),
            )
            orchestrator, _, _ = hosted_services()
            with st.spinner("Checking the image and available context..."):
                st.session_state["hosted_result"] = orchestrator.advise(request)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        finally:
            if temporary_upload and image_path is not None:
                image_path.unlink(missing_ok=True)

    result = st.session_state.get("hosted_result")
    if result is None:
        return

    st.subheader("Result")
    st.write(result.farmer_message)
    if result.metadata.get("market_is_synthetic"):
        st.warning("SIMULATED MARKET DATA — NOT LIVE MANDI PRICES.")
        with st.expander("Demonstration market context"):
            st.write(f"Market: {result.metadata.get('market_name', 'unavailable')}")
            st.write(f"Trend: {result.metadata.get('market_trend', 'unavailable')}")
            st.write(
                "Latest simulated modal price: "
                f"₹{result.metadata.get('market_latest_modal_price', 'unavailable')} per quintal"
            )
            st.write(
                "Simulated 7-day model forecast: "
                f"₹{result.metadata.get('market_predicted_7d_price', 'unavailable')} per quintal"
            )
    prediction = result.prediction
    if result.status == AdviceStatus.NEED_MORE_INFORMATION:
        st.metric("Calibrated image confidence", f"{result.metadata.get('confidence', 0):.0%}")
        st.caption("No disease label is released below the safety threshold.")
    if prediction is not None:
        st.metric("Calibrated image confidence", f"{prediction.confidence:.0%}")
        if result.status == AdviceStatus.NEED_MORE_INFORMATION:
            st.caption("No disease label is released below the safety threshold.")
        else:
            st.caption(f"Preliminary label: {prediction.display_name}")

    if result.status == AdviceStatus.PENDING_REVIEW:
        st.warning("This response is waiting for a qualified agronomist review.")
        st.code(result.review_id or "review-id-unavailable", language=None)
        if st.button("Refresh review status") and result.review_id:
            orchestrator, _, _ = hosted_services()
            try:
                st.session_state["hosted_result"] = orchestrator.public_review_status(
                    result.review_id
                )
                st.rerun()
            except KeyError:
                st.error("The review record is no longer available")

    if result.status in {
        AdviceStatus.READY,
        AdviceStatus.APPROVED,
        AdviceStatus.NEED_MORE_INFORMATION,
    }:
        helpful_col, unhelpful_col = st.columns(2)
        _, _, events = hosted_services()
        if helpful_col.button("Helpful", key=f"helpful_{result.request_id}"):
            events.log("farmer_feedback", result.request_id, helpful=True)
            st.success("Feedback recorded")
        if unhelpful_col.button("Not helpful", key=f"unhelpful_{result.request_id}"):
            events.log("farmer_feedback", result.request_id, helpful=False, reason="other")
            st.success("Feedback recorded")
    if result.status in {AdviceStatus.READY, AdviceStatus.APPROVED}:
        if st.button("Play response"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as handle:
                audio_output = Path(handle.name)
            try:
                GTTSService().synthesize(result.farmer_message, language, audio_output)
                st.audio(audio_output.read_bytes(), format="audio/mp3")
            except RuntimeError as exc:
                st.error(str(exc))
            finally:
                audio_output.unlink(missing_ok=True)


def render_reviewer() -> None:
    st.subheader("Agronomist review queue")
    st.warning("Only a qualified, authorized reviewer should approve advice.")
    configured_password = reviewer_password()
    if not configured_password:
        st.info("Reviewer access is disabled until REVIEWER_PASSWORD is configured.")
        return

    supplied_password = st.text_input("Reviewer password", type="password")
    if not secrets.compare_digest(supplied_password.encode(), configured_password.encode()):
        if supplied_password:
            st.error("Invalid reviewer password")
        return

    reviewer_name = st.text_input("Reviewer name / staff ID")
    _, store, events = hosted_services()
    queue = store.list_pending()
    st.metric("Pending cases", len(queue))
    for record in queue:
        title = f"{record.crop} · {record.disease} · {record.confidence:.0%}"
        with st.expander(title, expanded=len(queue) == 1):
            st.write("Farmer question:", record.question)
            st.write("Risk reasons:", record.risk.reasons)
            st.markdown("#### Evidence")
            for evidence in record.evidence:
                st.write(
                    f"**{evidence.source_id} — {evidence.title}** "
                    f"(expert validated: {evidence.expert_validated})"
                )
                st.caption(evidence.text)

            draft = record.private_draft
            summary = st.text_area("Summary", draft.summary, key=f"summary_{record.id}")
            actions = st.text_area(
                "Actions (one per line)",
                "\n".join(draft.actions),
                key=f"actions_{record.id}",
            )
            warnings = st.text_area(
                "Warning signs (one per line)",
                "\n".join(draft.warning_signs),
                key=f"warnings_{record.id}",
            )
            limitations = st.text_area(
                "Limitations", draft.limitations, key=f"limitations_{record.id}"
            )
            notes = st.text_area("Reviewer notes", key=f"notes_{record.id}")
            approve_col, reject_col = st.columns(2)
            if approve_col.button("Approve edited response", key=f"approve_{record.id}"):
                if not reviewer_name.strip():
                    st.error("Reviewer identity is required")
                else:
                    approved = AdviceDraft(
                        summary=summary,
                        actions=[line.strip() for line in actions.splitlines() if line.strip()],
                        warning_signs=[
                            line.strip() for line in warnings.splitlines() if line.strip()
                        ],
                        limitations=limitations,
                        sources=draft.sources,
                    )
                    updated = store.approve(record.id, reviewer_name, approved, notes)
                    events.log(
                        "review_decided",
                        updated.request_id,
                        review_id=updated.id,
                        decision=updated.status.value,
                    )
                    st.success("Approved")
                    st.rerun()
            if reject_col.button("Reject", key=f"reject_{record.id}"):
                if not reviewer_name.strip() or not notes.strip():
                    st.error("Reviewer identity and a rejection reason are required")
                else:
                    updated = store.reject(record.id, reviewer_name, notes)
                    events.log(
                        "review_decided",
                        updated.request_id,
                        review_id=updated.id,
                        decision=updated.status.value,
                    )
                    st.success("Rejected")
                    st.rerun()


def render_about() -> None:
    st.subheader("Model evidence and limitations")
    st.markdown(
        """
This public prototype uses EfficientNet-B0 fine-tuned on the official PlantDoc tomato subset.
After duplicate and conflicting-label quarantine, the split contains 552 training, 97 validation,
and 68 official test images across eight classes.

- Test accuracy: **52.9%**
- Test macro F1: **0.509**
- Coverage at the 0.65 abstention threshold: **26.5%**
- Accuracy among predictions above that threshold: **83.3%**
- Calibration error (ECE): **0.108**

These results do not establish field readiness. The dataset is small, classes are imbalanced,
external phone-photo generalization remains weak, and the public deployment uses ephemeral review
and feedback storage. It must not be used for pesticide dosage, legal compliance, poisoning
response, or a confirmed diagnosis.

**Supported labels:** healthy tomato leaf, early blight, late blight, Septoria leaf spot,
bacterial spot, leaf mold, mosaic virus, and yellow leaf curl virus.

The market table is generated synthetic demo data for seven Karnataka markets. It exists to
demonstrate feature engineering and advisory fusion; it is not a live price feed and must never be
used for a sale decision. Included walkthrough images come from the held-out official PlantDoc test
split and were not used for training.
"""
    )


st.title("🌿 Multilingual Crop Health Copilot")
farmer_tab, reviewer_tab, about_tab = st.tabs(["Farmer", "Reviewer", "Model card"])
with farmer_tab:
    render_farmer()
with reviewer_tab:
    try:
        render_reviewer()
    except ValueError:
        st.error("The case could not be updated. Refresh the queue and check your entries.")
with about_tab:
    render_about()

st.divider()
st.caption(
    "Educational decision support only. Low-confidence images are rejected and high-risk advice "
    "requires qualified human review."
)
