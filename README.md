Parallel Image Filtering using OpenCV + Streamlit

This project demonstrates parallel vs sequential image filtering using Python, OpenCV, and Streamlit.
It benchmarks processing time, visualizes speedup, efficiency, and compares theoretical vs measured performance using:

Amdahl’s Law

Gustafson’s Law

SPMD + Shared Memory Model

Perfect for Parallel & Distributed Computing (PDC) labs or understanding thread-level performance on CPUs.

✨ Features

✔️ Sequential vs Parallel image processing
✔️ Runs filters multiple times to simulate heavy computation
✔️ Adjustable threads (OpenCV multithreading)
✔️ Gaussian Blur, Average Blur, Sharpening & Edge Detection
✔️ Real-time metrics:

Execution time

Speedup

Efficiency
✔️ Amdahl vs Gustafson theoretical comparison plot
✔️ Interactive UI using Streamlit

📦 Installation
1️⃣ Clone Repository
git clone https://github.com/yourname/parallel-image-filtering.git
cd parallel-image-filtering

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install Required Libraries

Run:

pip install -r requirements.txt


Or install manually:

pip install streamlit opencv-python numpy plotly

▶️ Running the Application

Run Streamlit:

streamlit run image.py


Upload a large image (4K recommended) and test different:
✔️ Filters
✔️ Kernel sizes
✔️ Thread counts
✔️ Iteration levels

📁 Requirements File (requirements.txt)
streamlit
opencv-python
numpy
plotly


(Add this file to repo so users install everything with one command)

🧠 How It Works

OpenCV automatically divides the image into blocks

Each thread processes part of the image (domain decomposition)

Increasing threads reduces execution time — until overhead dominates

Amdahl/Gustafson laws predict speedup limits


🤝 Contributions

Pull requests are welcome!
Feel free to suggest improvements or add new filters.
