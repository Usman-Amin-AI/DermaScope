from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image

from src.image_quality import check_image_quality
from src.domain_gate import (
    CONFIDENT_DOMAIN_THRESHOLD,
    DEFAULT_DOMAIN_THRESHOLD,
    predict_domain,
)
from src.inference import predict
from src.gradcam import generate_gradcam, create_overlay
from src.reporting import build_batch_report_rows, build_csv_report


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_BATCH_IMAGES = 20
THEME_TYPE = getattr(st.context.theme, "type", None)
LOGO_NAME = (
    "dermascope_logo_dark.png"
    if THEME_TYPE == "dark"
    else "dermascope_logo.png"
)
LOGO_PATH = PROJECT_ROOT / "assets" / LOGO_NAME
MARK_PATH = PROJECT_ROOT / "assets" / "dermascope_mark.png"

st.set_page_config(
    page_title="DermaScope",
    page_icon=Image.open(MARK_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.logo(str(LOGO_PATH))

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin-bottom: .25rem;}
    .hero p {margin-bottom: 0; opacity: .82;}
    .notice {
        padding: .85rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,193,7,.45);
        background: rgba(255,193,7,.08);
    }
    .result-card {
        padding: 1.1rem 1.2rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
    }
    .small {font-size: .86rem; opacity: .72;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_NAMES = [
    "Actinic keratoses / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


# ---------- Header ----------
st.image(str(LOGO_PATH), width=340)
st.markdown(
    """
    <div class="hero">
        <p>Educational classification of dermoscopic skin-lesion images.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice"><strong>Important:</strong> This is an educational '
    'AI prototype, not a medical diagnosis or a substitute for a qualified clinician. '
    'The model was trained and evaluated on dermoscopic images from DermaMNIST.</div>',
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Analysis settings")
    upload_mode = st.segmented_control(
        "Upload mode",
        options=["Single image", "Batch images"],
        default="Single image",
        key="upload_mode",
    )

    st.divider()
    threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Predictions below this softmax threshold are presented as uncertain.",
    )

    st.divider()
    st.subheader("Model")
    st.write("**EfficientNet-B0 V4**")
    st.write("Input: **224 × 224**")
    st.write("Dataset: **DermaMNIST**")
    st.write("Task: **7-class classification**")

    st.divider()
    st.caption(
        "Softmax confidence is a model score, not a clinically calibrated probability."
    )


# ---------- Batch input and results ----------
if upload_mode == "Batch images":
    st.subheader("1. Upload dermoscopic images")
    uploaded_files = st.file_uploader(
        "Choose multiple JPG, JPEG, or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_image_upload",
        help="Images are checked individually. The domain score is advisory and does not block analysis.",
    )

    if not uploaded_files:
        st.info("Select two or more images to begin a batch analysis.")
        st.stop()

    if len(uploaded_files) > MAX_BATCH_IMAGES:
        st.error(
            f"Batch analysis supports a maximum of {MAX_BATCH_IMAGES} images per run."
        )
        st.stop()

    oversized_files = [
        uploaded_file.name
        for uploaded_file in uploaded_files
        if uploaded_file.size > MAX_FILE_SIZE_BYTES
    ]
    if oversized_files:
        st.error(
            "Each image must be 20 MB or smaller. "
            f"Oversized file(s): {', '.join(oversized_files)}"
        )
        st.stop()

    upload_signature = (
        threshold,
        tuple(
            (uploaded_file.name, hashlib.sha256(uploaded_file.getvalue()).hexdigest())
            for uploaded_file in uploaded_files
        ),
    )
    cached_batch = st.session_state.get("batch_results")
    if cached_batch and cached_batch["signature"] != upload_signature:
        st.session_state.pop("batch_results", None)
        cached_batch = None

    st.caption(f"{len(uploaded_files)} image(s) selected")

    preview_files = uploaded_files[: min(6, len(uploaded_files))]
    preview_columns = st.columns(min(3, len(preview_files)))
    for index, uploaded_file in enumerate(preview_files):
        try:
            preview = Image.open(uploaded_file).convert("RGB")
            with preview_columns[index % len(preview_columns)]:
                st.image(
                    preview,
                    caption=f"{index + 1}. {uploaded_file.name}",
                    width=180,
                )
            uploaded_file.seek(0)
        except Exception:
            st.warning(f"Could not preview {uploaded_file.name}.")

    analyze_batch = st.button(
        "Analyze batch",
        type="primary",
        icon=":material/analytics:",
        key="analyze_batch",
    )

    if analyze_batch:
        rows = []
        details = {}
        progress = st.progress(0, text="Preparing batch analysis...")

        for index, uploaded_file in enumerate(uploaded_files):
            file_label = f"{index + 1}. {uploaded_file.name}"
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                rows.append({
                    "Filename": file_label,
                    "Quality": "Unreadable",
                    "Domain score": "Not run",
                    "Prediction": "Not run",
                    "Confidence": "Not run",
                    "Decision": "File could not be read",
                })
                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Checked {index + 1} of {len(uploaded_files)} images",
                )
                continue

            quality = check_image_quality(image)
            if not quality.passed:
                rows.append({
                    "Filename": file_label,
                    "Quality": "Needs review",
                    "Domain score": "Not run",
                    "Prediction": "Not run",
                    "Confidence": "Not run",
                    "Decision": "; ".join(quality.reasons),
                })
                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Checked {index + 1} of {len(uploaded_files)} images",
                )
                continue

            try:
                domain = predict_domain(
                    image,
                    threshold=DEFAULT_DOMAIN_THRESHOLD,
                )
                result = predict(image, uncertainty_threshold=threshold)
            except Exception as exc:
                rows.append({
                    "Filename": file_label,
                    "Quality": "Passed",
                    "Domain score": "Not available",
                    "Prediction": "Not available",
                    "Confidence": "Not available",
                    "Decision": f"Analysis failed: {exc}",
                })
                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Checked {index + 1} of {len(uploaded_files)} images",
                )
                continue

            domain_score = float(domain["dermoscopic_probability"])
            decision = result["status"]
            if domain_score < CONFIDENT_DOMAIN_THRESHOLD:
                decision = f"{decision}; domain uncertain"

            rows.append({
                "Filename": file_label,
                "Quality": "Passed",
                "Domain score": f"{domain_score:.1%}",
                "Prediction": result["prediction"],
                "Confidence": f"{result['confidence']:.1%}",
                "Decision": decision,
            })
            details[file_label] = {
                "image": image,
                "result": result,
                "domain_score": domain_score,
            }
            progress.progress(
                (index + 1) / len(uploaded_files),
                text=f"Checked {index + 1} of {len(uploaded_files)} images",
            )

        progress.empty()
        st.session_state["batch_results"] = {
            "signature": upload_signature,
            "rows": rows,
            "details": details,
        }

    cached_batch = st.session_state.get("batch_results")
    if cached_batch and cached_batch["signature"] == upload_signature:
        rows = cached_batch["rows"]
        details = cached_batch["details"]
        analyzed_count = len(details)
        quality_issues = sum(row["Quality"] != "Passed" for row in rows)

        st.subheader("2. Batch results")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Images", len(rows))
        metric_columns[1].metric("Analyzed", analyzed_count)
        metric_columns[2].metric("Needs attention", quality_issues)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download results report",
            data=build_csv_report(build_batch_report_rows(rows, details, threshold)),
            file_name="dermascope_batch_report.csv",
            mime="text/csv",
            icon=":material/download:",
            key="batch_report_download",
        )

        if details:
            selected_name = st.selectbox(
                "View image details",
                options=list(details),
                key="batch_detail_image",
            )
            selected = details[selected_name]
            selected_result = selected["result"]

            detail_columns = st.columns([1, 1.2])
            with detail_columns[0]:
                st.image(
                    selected["image"],
                    caption=selected_name,
                    width=320,
                )
                st.metric("Predicted class", selected_result["prediction"])
                st.metric("Model confidence", f"{selected_result['confidence']:.1%}")
            with detail_columns[1]:
                if selected["domain_score"] < CONFIDENT_DOMAIN_THRESHOLD:
                    st.warning(
                        "The domain score is uncertain. Genuine dermoscopic images can score low; "
                        "the score did not block classification."
                    )
                if selected_result["status"] == "Uncertain":
                    st.warning(
                        f"Model confidence is below the selected {threshold:.0%} threshold."
                    )
                ordered = sorted(
                    selected_result["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                st.bar_chart(
                    {name: float(value) for name, value in ordered},
                    horizontal=True,
                )

            if st.button(
                "Generate Grad-CAM for selected image",
                icon=":material/center_focus_strong:",
                key="batch_gradcam",
            ):
                try:
                    with st.spinner("Generating Grad-CAM..."):
                        heatmap, target_class, cam_confidence = generate_gradcam(
                            selected["image"],
                            target_class=None,
                        )
                        overlay = create_overlay(selected["image"], heatmap)
                    st.image(
                        overlay,
                        caption=(
                            f"Grad-CAM — {CLASS_NAMES[target_class]} "
                            f"({cam_confidence:.1%})"
                        ),
                        width=420,
                    )
                except Exception as exc:
                    st.warning(f"Grad-CAM could not be generated: {exc}")

        st.caption(
            "Educational output only. Domain scores and softmax confidence are not clinical probabilities."
        )

    st.stop()


# ---------- Single-image input and results ----------
st.subheader("1. Upload dermoscopic image")

uploaded = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
    help="For meaningful evaluation, use a dermoscopic skin-lesion image similar to the training domain.",
)

if uploaded is None:
    st.info("Upload an image to begin the analysis.")
    st.stop()

if uploaded.size > MAX_FILE_SIZE_BYTES:
    st.error("Each image must be 20 MB or smaller.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.image(image, caption="Uploaded image", width="stretch")

with right:
    st.subheader("Image quality")

    quality = check_image_quality(image)

    if quality.passed:
        st.success(f"Quality check passed — score {quality.score:.2f}")
    else:
        st.warning(f"Quality check needs attention — score {quality.score:.2f}")
        for reason in quality.reasons:
            st.write(f"• {reason}")

    st.caption(
        "Quality checks help prevent obviously unsuitable inputs from reaching the model."
    )

st.divider()

# ---------- Analysis ----------
st.subheader("2. Run AI analysis")

if not quality.passed:
    st.error("Analysis stopped because the image did not pass the quality check.")
    st.stop()

# ---------- Dermoscopic domain validation ----------
with st.spinner("Validating image domain..."):
    domain = predict_domain(
        image,
        threshold=DEFAULT_DOMAIN_THRESHOLD,
    )

domain_probability = domain["dermoscopic_probability"]

if domain_probability >= CONFIDENT_DOMAIN_THRESHOLD:
    st.success(
        "Input-domain check passed — image accepted for skin-lesion classification."
    )
else:
    st.warning(
        "The domain check is uncertain and can misclassify genuine dermoscopic images. "
        "You can still run the analysis; results are unreliable for images outside "
        "the dermoscopic training domain."
    )

    with st.expander("Input-domain validation details"):
        st.write(
            {
                "dermoscopic_probability": domain_probability,
                "non_dermoscopic_probability": domain["non_dermoscopic_probability"],
                "advisory_threshold": domain["threshold"],
                "high_confidence_threshold": CONFIDENT_DOMAIN_THRESHOLD,
            }
        )

    st.caption("The domain score does not prevent you from analyzing the image.")


analyze = st.button(
    "Analyze Image",
    type="primary",
    width="stretch",
)

if not analyze:
    st.stop()

with st.spinner("Running EfficientNet-B0 V4 analysis..."):
    result = predict(image, uncertainty_threshold=threshold)

prediction = result["prediction"]
confidence = float(result["confidence"])
status = result["status"]
probabilities = result["probabilities"]

if status != "Prediction available":
    st.warning(f"Result: {status}")
else:
    st.success("Prediction available")

# ---------- Result ----------
st.subheader("3. Classification result")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("Predicted class", prediction)

with r2:
    st.metric("Model confidence", f"{confidence:.1%}")

with r3:
    st.metric("Decision", "Accepted" if confidence >= threshold else "Uncertain")

if confidence < threshold:
    st.warning(
        f"The model confidence ({confidence:.1%}) is below the selected "
        f"threshold ({threshold:.0%}). Treat this result as uncertain."
    )

st.markdown(
    '<p class="small">The prediction is an AI model output and should not be interpreted '
    'as a definitive medical diagnosis.</p>',
    unsafe_allow_html=True,
)

# ---------- Probability distribution ----------
with st.expander("View class probability distribution", expanded=True):
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    chart_data = {name: float(value) for name, value in ordered}
    st.bar_chart(chart_data, horizontal=True)

single_report = {
    "Model": "EfficientNet-B0 V4",
    "Filename": uploaded.name,
    "Quality": "Passed",
    "Quality score": quality.score,
    "Domain score": domain_probability,
    "Prediction": prediction,
    "Confidence": confidence,
    "Uncertainty threshold": threshold,
    "Decision": status,
    **{f"Probability: {name}": value for name, value in probabilities.items()},
}
st.download_button(
    "Download results report",
    data=build_csv_report([single_report]),
    file_name="dermascope_image_report.csv",
    mime="text/csv",
    icon=":material/download:",
    key="single_report_download",
)

# ---------- Explainability ----------
st.divider()
st.subheader("4. Model explainability")

st.write(
    "Grad-CAM highlights image regions that contributed to the selected model output. "
    "It is an AI explainability visualization, not a clinical finding or lesion boundary."
)

try:
    with st.spinner("Generating Grad-CAM explanation..."):
        heatmap, target_class, cam_confidence = generate_gradcam(
            image,
            target_class=None,
        )
        overlay = create_overlay(image, heatmap)

    c1, c2 = st.columns(2)

    with c1:
        st.image(image, caption="Original image", width="stretch")

    with c2:
        st.image(
            overlay,
            caption=f"Grad-CAM — {CLASS_NAMES[target_class]} ({cam_confidence:.1%})",
            width="stretch",
        )

except Exception as exc:
    st.warning(f"Grad-CAM could not be generated for this image: {exc}")

# ---------- Technical details ----------
with st.expander("Technical details"):
    st.write(
        {
            "model": "EfficientNet-B0 V4",
            "input_size": "224x224",
            "dataset": "DermaMNIST",
            "classes": 7,
            "uncertainty_threshold": threshold,
            "confidence": confidence,
            "status": status,
        }
    )

# ---------- Footer ----------
st.divider()
st.caption(
    "DermaScope • EfficientNet-B0 V4 • DermaMNIST • Educational prototype • "
    "V4 frozen release"
)
