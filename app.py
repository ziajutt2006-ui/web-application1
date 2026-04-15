import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import time

# --- Page Config ------------------------------------------------------------
st.set_page_config(
    page_title="VisionAI · Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* -- Root & Body -- */
:root {
    --bg:        #0a0a0f;
    --bg2:       #12121a;
    --bg3:       #1a1a26;
    --accent:    #7c6af7;
    --accent2:   #a78bfa;
    --lime:      #b6f542;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --border:    rgba(124,106,247,.22);
    --glow:      0 0 28px rgba(124,106,247,.35);
}

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* -- Hide Streamlit chrome -- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

/* -- Hero header -- */
.hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.2rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
}
.hero-icon {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: var(--glow);
    flex-shrink: 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #fff 0%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin: 0;
}
.hero-sub {
    font-size: .72rem; color: var(--muted); letter-spacing: .12em;
    text-transform: uppercase; margin-top: .3rem;
}

/* -- Upload zone -- */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Mono', monospace !important;
    color: var(--muted) !important;
}

/* -- Metric cards -- */
.metric-row { display: flex; gap: 1rem; margin: 1.6rem 0 1.2rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    display: flex; flex-direction: column; gap: .3rem;
}
.metric-label { font-size: .62rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
.metric-value {
    font-family: 'Syne', sans-serif; font-size: 1.65rem; font-weight: 800;
    color: var(--lime); line-height: 1;
}
.metric-sub { font-size: .65rem; color: var(--muted); }

/* -- Detection table -- */
.det-table { width: 100%; border-collapse: collapse; font-size: .78rem; }
.det-table th {
    background: var(--bg3);
    color: var(--muted);
    font-weight: 500; letter-spacing: .08em; text-transform: uppercase;
    padding: .55rem 1rem; text-align: left;
    border-bottom: 1px solid var(--border);
}
.det-table td {
    padding: .6rem 1rem;
    border-bottom: 1px solid rgba(124,106,247,.08);
    color: var(--text);
}
.det-table tr:hover td { background: rgba(124,106,247,.06); }
.conf-bar-bg { width: 80px; height: 6px; background: var(--bg3); border-radius: 3px; display: inline-block; vertical-align: middle; margin-left: .5rem; }
.conf-bar    { height: 6px; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--lime)); }
.badge {
    display: inline-block;
    background: rgba(124,106,247,.15);
    color: var(--accent2);
    border: 1px solid rgba(124,106,247,.3);
    border-radius: 6px;
    padding: .15rem .5rem;
    font-size: .68rem; letter-spacing: .06em;
}

/* -- Section label -- */
.sec-label {
    font-family: 'Syne', sans-serif; font-size: .72rem;
    text-transform: uppercase; letter-spacing: .15em;
    color: var(--muted); margin: 1.6rem 0 .8rem;
    display: flex; align-items: center; gap: .6rem;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* -- Sidebar pills -- */
.sb-pill {
    display: inline-block;
    background: rgba(124,106,247,.12);
    border: 1px solid var(--border);
    color: var(--accent2);
    border-radius: 20px;
    padding: .2rem .7rem;
    font-size: .65rem; letter-spacing: .06em;
    margin: .15rem;
}
.sb-step {
    display: flex; gap: .7rem; align-items: flex-start;
    margin-bottom: .8rem; font-size: .75rem; color: var(--text);
}
.sb-num {
    width: 22px; height: 22px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .6rem; font-weight: 700; color: #fff;
}

/* -- Slider & button -- */
.stSlider > div > div > div { background: var(--accent) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; letter-spacing: .06em !important;
    padding: .55rem 1.6rem !important;
    box-shadow: var(--glow) !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* -- Image frame -- */
.img-frame {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 40px rgba(0,0,0,.45);
}
</style>
""", unsafe_allow_html=True)

# --- Lazy model load ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # ~6 MB, auto-downloaded on first run

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='padding:.6rem 0 1rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;
                  background:linear-gradient(90deg,#fff,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        VisionAI
      </div>
      <div style='font-size:.62rem;color:#6b6b85;letter-spacing:.12em;text-transform:uppercase'>
        Object Detection · YOLOv8n
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Instructions
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:.75rem;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem'>How to use</div>", unsafe_allow_html=True)
    for n, step in enumerate([
        "Upload any JPG / PNG image",
        "Adjust confidence threshold",
        "Click **Detect Objects**",
        "View annotated results & table",
    ], 1):
        st.markdown(f"""
        <div class='sb-step'>
          <div class='sb-num'>{n}</div>
          <div>{step}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Confidence slider
    st.markdown("<div style='font-size:.65rem;color:#6b6b85;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem'>Confidence threshold</div>", unsafe_allow_html=True)
    conf_threshold = st.slider("", min_value=0.10, max_value=0.95, value=0.35, step=0.05, label_visibility="collapsed")
    st.markdown(f"<div style='font-size:.7rem;color:#b6f542;text-align:right;margin-top:-.4rem'>{int(conf_threshold*100)}%</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Model info
    st.markdown("<div style='font-family:Syne,sans-serif;font-size:.75rem;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.6rem'>Model info</div>", unsafe_allow_html=True)
    for pill in ["YOLOv8n · Ultralytics", "80 COCO classes", "~6 MB weights", "100% Free · No API"]:
        st.markdown(f"<span class='sb-pill'>{pill}</span>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;font-size:.65rem;color:#6b6b85'>Detects: people, cars, animals, furniture, food, electronics, and 74 more COCO categories.</div>", unsafe_allow_html=True)

# --- Main area ----------------------------------------------------------------

# Hero
st.markdown("""
<div class='hero'>
  <div class='hero-icon'>🔍</div>
  <div>
    <div class='hero-title'>Object Detection</div>
    <div class='hero-sub'>YOLOv8n · Real-time · Zero cost</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Upload
st.markdown("<div class='sec-label'>Upload Image</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp", "bmp"], label_visibility="collapsed")

if uploaded is None:
    # Empty state
    st.markdown("""
    <div style='text-align:center;padding:3.5rem 1rem;
                background:#12121a;border:1.5px dashed rgba(124,106,247,.2);
                border-radius:20px;margin-top:.5rem'>
      <div style='font-size:3rem;margin-bottom:.8rem'>🖼️</div>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#e8e8f0'>
        Drop an image to get started
      </div>
      <div style='font-size:.75rem;color:#6b6b85;margin-top:.5rem'>
        Supports JPG · PNG · WEBP · BMP
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# -- Load image --
pil_image = Image.open(uploaded).convert("RGB")
img_array = np.array(pil_image)

col_orig, col_gap, col_result = st.columns([1, 0.04, 1])

with col_orig:
    st.markdown("<div class='sec-label'>Original</div>", unsafe_allow_html=True)
    st.image(pil_image, use_container_width=True)

# -- Detect button --
st.markdown("<div style='margin:1rem 0 .5rem'>", unsafe_allow_html=True)
run = st.button("⚡  Detect Objects", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

if not run:
    with col_result:
        st.markdown("<div class='sec-label'>Annotated Result</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;align-items:center;justify-content:center;
                    height:200px;background:#12121a;border-radius:16px;
                    border:1px dashed rgba(124,106,247,.18);color:#6b6b85;font-size:.78rem'>
          Results will appear here
        </div>""", unsafe_allow_html=True)
    st.stop()

# -- Run inference --
with st.spinner("Loading model & running inference…"):
    model = load_model()
    t0 = time.time()
    results = model(img_array, conf=conf_threshold, verbose=False)
    elapsed = (time.time() - t0) * 1000   # ms

result  = results[0]
boxes   = result.boxes
n_dets  = len(boxes)

# -- Annotate image WITH PIL (No OpenCV needed) --
annotated_pil = pil_image.copy()
draw = ImageDraw.Draw(annotated_pil)

# Load default font for drawing text
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

colors_cache = {}
def class_color(cls_id):
    if cls_id not in colors_cache:
        rng = np.random.default_rng(cls_id * 97 + 13)
        colors_cache[cls_id] = tuple(int(x) for x in rng.integers(80, 230, 3))
    return colors_cache[cls_id]

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf  = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    color = class_color(cls_id)

    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw label tag
    tag = f"{label} {conf:.0%}"
    
    # Handle text size dynamically
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(tag)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = draw.textsize(tag, font=font)
        
    draw.rectangle([x1, y1 - text_height - 6, x1 + text_width + 8, y1], fill=color)
    draw.text((x1 + 4, y1 - text_height - 4), tag, fill=(255, 255, 255), font=font)

with col_result:
    st.markdown("<div class='sec-label'>Annotated Result</div>", unsafe_allow_html=True)
    st.image(annotated_pil, use_container_width=True)

# -- Metrics --
unique_classes = len(set(int(b.cls[0]) for b in boxes)) if n_dets else 0
avg_conf = (sum(float(b.conf[0]) for b in boxes) / n_dets * 100) if n_dets else 0

st.markdown(f"""
<div class='metric-row'>
  <div class='metric-card'>
    <div class='metric-label'>Objects found</div>
    <div class='metric-value'>{n_dets}</div>
    <div class='metric-sub'>detections</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Unique classes</div>
    <div class='metric-value'>{unique_classes}</div>
    <div class='metric-sub'>categories</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Avg confidence</div>
    <div class='metric-value'>{avg_conf:.0f}%</div>
    <div class='metric-sub'>mean score</div>
  </div>
  <div class='metric-card'>
    <div class='metric-label'>Inference time</div>
    <div class='metric-value'>{elapsed:.0f}</div>
    <div class='metric-sub'>milliseconds</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -- Detection table --
if n_dets == 0:
    st.markdown("""
    <div style='text-align:center;padding:2rem;background:#12121a;border-radius:14px;
                border:1px solid rgba(124,106,247,.15);color:#6b6b85;font-size:.78rem'>
      No objects detected above the confidence threshold.<br>
      Try lowering the slider in the sidebar.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("<div class='sec-label'>Detection Details</div>", unsafe_allow_html=True)

    det_list = sorted([
        (model.names[int(b.cls[0])], float(b.conf[0]),
         list(map(int, b.xyxy[0].tolist())), int(b.cls[0]))
        for b in boxes
    ], key=lambda x: -x[1])

    rows_html = ""
    for i, (label, conf, bbox, cls_id) in enumerate(det_list, 1):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        bar_w = int(conf * 80)
        rows_html += f"""
        <tr>
          <td style='color:#6b6b85'>{i:02d}</td>
          <td><span class='badge'>{label}</span></td>
          <td>
            {conf:.1%}
            <span class='conf-bar-bg'><span class='conf-bar' style='width:{bar_w}px;display:block'></span></span>
          </td>
          <td style='color:#6b6b85;font-size:.7rem'>{w}×{h} px</td>
          <td style='color:#6b6b85;font-size:.7rem'>{bbox[0]},{bbox[1]}</td>
        </tr>"""

    st.markdown(f"""
    <table class='det-table'>
      <thead><tr>
        <th>#</th><th>Class</th><th>Confidence</th><th>Size</th><th>Top-left</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # -- Download annotated image --
    st.markdown("<div style='margin-top:1.4rem'>", unsafe_allow_html=True)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇  Download Annotated Image",
        data=buf.getvalue(),
        file_name="detected_objects.png",
        mime="image/png",
    )
    st.markdown("</div>", unsafe_allow_html=True)
