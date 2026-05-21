# Solar Explorar — AI Rooftop Analytics

> **Empowering the Solar Revolution with Deep Learning.**  
> *A high-performance roof segmentation engine that calculates solar potential from satellite imagery in real-time.*

---

## Problem Statement
As the world transitions to renewable energy, many homeowners and businesses are unaware of their own rooftop's solar potential. Manual assessment is slow, expensive, and non-scalable. 

**Solar Explorar** solves this by using **Computer Vision (U-Net)** to automatically segment rooftops from satellite data, calculate usable surface area, and provide an instant estimate of energy production, financial savings, and CO2 offset.

##  Global Impact: Alignment with SDGs
Solar Explorar is built with a commitment to the United Nations **Sustainable Development Goals (SDGs)**:

- **SDG 7: Affordable and Clean Energy**  
  Democratizing solar energy access by providing free, instant assessment tools to help homeowners transition to renewables.
- **SDG 11: Sustainable Cities and Communities**  
  Driving urban sustainability through city-wide rooftop analysis and optimized solar deployment.
- **SDG 13: Climate Action**  
  Directly combatting carbon emissions by quantifying the CO2 offset potential of dormant rooftop spaces.

##  Key Features
- **Precise Segmentation**: Custom U-Net ensemble (ResNet34 backbone) trained for high-accuracy binary mask generation.
- **Satellite Integration**: Real-time imagery fetching via **Mapbox Static API**.
- **Smart Analytics**: Automatic calculation of:
  - **Usable Roof Area (m²)**
  - **Estimated Solar Panel Count**
  - **System Capacity (kW)**
  - **Annual Energy Generation (kWh)**
  - **Financial Savings (Rs)**
- **Professional Dashboard**: Clean, modern UI for instant visualization of analysis results and overlays.
- **Demo Mode**: Built-in simulator for offline presentations and cost-effective testing.

##  Tech Stack
- **Frontend**: HTML5, Vanilla CSS3 (Modern UI/UX), JavaScript (ES6+).
- **Backend**: Python, **FastAPI**, Uvicorn.
- **AI/ML**: **PyTorch**, Segmentation Models PyTorch (SMP), Albumentations (Augmentation), OpenCV.
- **Data/APIs**: Mapbox (Geocoding & Satellite Imagery).

---

##  Architecture
1. **Input**: User enters an address or coordinates.
2. **Geocoding**: Mapbox converts text to precise Lat/Lng.
3. **Imagery**: High-resolution satellite tiles are retrieved.
4. **Inference**: A K-Fold Ensemble of **U-Net** models processes the image.
5. **Post-Processing**: OpenCV extracts contours and calculates pixel-to-meter area scaling.
6. **Output**: Interactive dashboard displays the segmentation overlay and solar metrics.

---

##  Getting Started

### Prerequisites
- Python 3.9+
- [Mapbox Access Token](https://www.mapbox.com/help/how-mapbox-data-works/) (Optional, dummy mode available)

### Installation
1. **Clone the repo**
   ```bash
   git clone https://github.com/IppiliSahasra08/solar_explorar.git
   cd solar_explorar
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment** (Optional)
   Create a `config.py` or set environment variables:
   ```python
   MAPBOX_TOKEN = "your_token_here"
   ```

4. **Launch the Application**
   ```bash
   # Start the Backend
   python app.py
   
   # Open index.html in your browser
   ```

---

##  Meet the Team
- **Ippili Sahasra** - 24BCE5035
- **Sri Poojitha Sudalagunta** - 24BCE1637
- **Shreya Kailash** - 24BCE5513
- **Ananya Arumbakkam** - 24BCE5154

