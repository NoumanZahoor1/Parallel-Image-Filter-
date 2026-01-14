import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import os

# ==================== AMDAHL & GUSTAFSON LAWS ====================
def amdahl_speedup(n, f):
    """Amdahl's Law: Speedup with fixed problem size"""
    if f >= 1.0:
        return 1.0
    if f <= 0.0:
        return float(n)
    return 1.0 / (f + (1.0 - f) / n)

def gustafson_speedup(n, f):
    """Gustafson's Law: Speedup with scaled problem size"""
    if f >= 1.0:
        return 1.0
    if f <= 0.0:
        return float(n)
    return n - f * (n - 1)

# ==================== FILTER FUNCTION ====================
def apply_filter(img, filter_type, kernel):
    """
    Applies the selected image filter.
    NOTE:
    - This function is executed by multiple threads in parallel
    - OpenCV internally partitions the image into blocks
    - Each thread processes a different block (Domain Decomposition)
    """
    if filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(img, (kernel, kernel), 0)

    elif filter_type == "Blur":
        return cv2.blur(img, (kernel, kernel))

    elif filter_type == "Sharpen":
        sharpen_kernel = np.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]], dtype=np.float32
        )
        return cv2.filter2D(img, -1, sharpen_kernel)

    else:  # Edge Detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# ==================== UI ====================
st.set_page_config(
    page_title="PDC Lab - Parallel Image Filtering",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🚀 Parallel Image Filtering using OpenCV Threads</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>SPMD + Shared Memory Model | Performance Analysis using Amdahl & Gustafson</p>",
    unsafe_allow_html=True
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded = st.file_uploader(
        "Upload Large Image (4K / 8K recommended)",
        type=["jpg", "png", "jpeg"]
    )

    filter_type = st.selectbox(
        "Filter Type",
        ["Gaussian Blur", "Blur", "Sharpen", "Edge Detection"]
    )

    kernel = st.selectbox(
        "Kernel Size (Higher = More Computation)",
        [21, 15, 11, 9, 7, 5, 3],
        index=0
    )

    iterations = st.slider(
        "Computation Intensity (Iterations)",
        1, 20, 5
    )

    max_threads = min(16, os.cpu_count() or 8)
    threads = st.slider(
        "Number of Threads",
        1, max_threads, max_threads
    )

    st.caption(f"🖥 CPU Cores Detected: {os.cpu_count()}")

# ==================== MAIN PROCESSING ====================
if uploaded:
    # Load image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    st.image(img_rgb, caption=f"Original Image — {w}×{h}px", use_column_width=True)

    if st.button("🚀 Run Sequential vs Parallel", use_container_width=True):

        progress = st.progress(0)
        status = st.empty()

        # ==================== SEQUENTIAL ====================
        status.text("Running Sequential Version (1 Thread)...")
        cv2.setNumThreads(1)

        seq_img = img_rgb.copy()
        start = time.perf_counter()

        for _ in range(iterations):
            seq_img = apply_filter(seq_img, filter_type, kernel)

        seq_time = time.perf_counter() - start
        progress.progress(50)

        # ==================== PARALLEL ====================
        status.text(f"Running Parallel Version ({threads} Threads)...")
        cv2.setNumThreads(threads)

        par_img = img_rgb.copy()
        start = time.perf_counter()

        for _ in range(iterations):
            par_img = apply_filter(par_img, filter_type, kernel)

        par_time = time.perf_counter() - start
        progress.progress(100)

        # ==================== METRICS ====================
        speedup = seq_time / par_time if par_time > 0 else 1.0
        efficiency = (speedup / threads) * 100 if threads > 0 else 0.0

        serial_frac = (
            (1/speedup - 1/threads) / (1 - 1/threads)
            if threads > 1 and speedup > 1 else 0.05
        )

        # ==================== RESULTS ====================
        col1, col2 = st.columns(2)
        with col1:
            st.image(seq_img, caption=f"Sequential\nTime: {seq_time:.4f}s", use_column_width=True)
        with col2:
            st.image(par_img, caption=f"Parallel ({threads} Threads)\nTime: {par_time:.4f}s", use_column_width=True)

        st.markdown("### 📊 Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequential Time", f"{seq_time:.4f}s")
        c2.metric("Parallel Time", f"{par_time:.4f}s")
        c3.metric("Speedup", f"{speedup:.2f}×")
        c4.metric("Efficiency", f"{efficiency:.1f}%")

        # ==================== THEORY PLOT ====================
        st.markdown("### 📈 Amdahl’s vs Gustafson’s Law")

        x = np.linspace(1, 32, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=[amdahl_speedup(p, serial_frac) for p in x],
            name="Amdahl's Law",
            line=dict(color="red")
        ))
        fig.add_trace(go.Scatter(
            x=x, y=[gustafson_speedup(p, serial_frac) for p in x],
            name="Gustafson's Law",
            line=dict(color="green", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=[threads], y=[speedup],
            mode="markers+text",
            name="Measured Speedup",
            marker=dict(size=12, color="blue"),
            text=[f"{speedup:.2f}×"],
            textposition="top center"
        ))

        fig.update_layout(
            xaxis_title="Number of Threads",
            yaxis_title="Speedup",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        if speedup > 1.2:
            st.success(f"✅ Achieved {speedup:.2f}× speedup using parallel execution!")
            st.balloons()
        else:
            st.info("ℹ️ Increase image size, kernel, or iterations for better speedup.")

        status.empty()
        progress.empty()

else:
    st.info("👈 Upload a high-resolution image to start parallel processing.")
    st.markdown("""
    **Tips for best results:**
    - Use large images (4K or higher)
    - Choose large kernel sizes
    - Increase iterations
    - Match threads with CPU cores
    """)
