const API = 'http://localhost:8000';

// ---------- OVAL REVEAL ----------
const transition = document.getElementById("analyze");
const panel = document.getElementById("panel");

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const easeInOut = (x) => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;

function updateReveal() {
  const rect = transition.getBoundingClientRect();
  const viewH = window.innerHeight;

  const total = rect.height - viewH;
  const scrolled = -rect.top;
  const t = total > 0 ? clamp01(scrolled / total) : 1;
  const e = easeInOut(t);

  const w = window.innerWidth;
  const h = window.innerHeight;
  const diag = Math.sqrt(w * w + h * h);

  const startRx = 80, startRy = 150;
  const endRx = diag * 0.92, endRy = diag * 0.92;

  const rx = startRx + (endRx - startRx) * e;
  const ry = startRy + (endRy - startRy) * e;

  const cx = 50, cy = 40;
  panel.style.clipPath = `ellipse(${rx}px ${ry}px at ${cx}% ${cy}%)`;
  panel.style.setProperty("--shade", (1 - e) * 0.85);

  requestAnimationFrame(updateReveal);
}
requestAnimationFrame(updateReveal);

// ---------- REQUIRED CALCULATIONS ----------
const AREA_PER_PANEL_M2 = 8;
const PANEL_WATTS = 470;
const PANEL_KW = PANEL_WATTS / 1000;

const ASSUMED_MONTHLY_KWH_MIN = 300;
const ASSUMED_MONTHLY_KWH_MAX = 400;
const ASSUMED_MONTHLY_KWH_AVG = 350;

const PRICE_PER_KWH_USD = 0.15;
const CO2_KG_PER_KWH = 0.82;
const KWH_PER_KW_PER_YEAR = 1460;

function computeMetricsFromRoofArea(roofAreaM2) {
  const panels = Math.max(1, Math.floor(roofAreaM2 / AREA_PER_PANEL_M2));
  const capacityKw = panels * PANEL_KW;

  const assumedAnnualKwh = ASSUMED_MONTHLY_KWH_AVG * 12;
  const estAnnualGenKwh = capacityKw * KWH_PER_KW_PER_YEAR;
  const offsetKwh = Math.min(assumedAnnualKwh, estAnnualGenKwh);

  const annualSavings = offsetKwh * PRICE_PER_KWH_USD;
  const co2OffsetKg = offsetKwh * CO2_KG_PER_KWH;

  return {
    roof_area_m2: +roofAreaM2.toFixed(2),
    panel_count: panels,
    capacity_kw: +capacityKw.toFixed(2),
    annual_savings: Math.round(annualSavings),
    co2_offset_kg: Math.round(co2OffsetKg),
    assumed_monthly_kwh_avg: ASSUMED_MONTHLY_KWH_AVG,
    assumed_monthly_kwh_range: [ASSUMED_MONTHLY_KWH_MIN, ASSUMED_MONTHLY_KWH_MAX],
    est_annual_gen_kwh: Math.round(estAnnualGenKwh),
    offset_kwh: Math.round(offsetKwh),
  };
}

// ---------- APP LOGIC ----------
function switchTab(mode) {
  document.getElementById('tab-address').classList.toggle('active', mode === 'address');
  document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
  document.getElementById('address-panel').style.display = mode === 'address' ? 'flex' : 'none';
  document.getElementById('upload-zone').style.display = mode === 'upload' ? 'block' : 'none';
  hideDashboard();
}

function setLoading(on, msg = 'Processing...') {
  document.getElementById('loader').style.display = on ? 'block' : 'none';
  document.getElementById('loader-text').style.display = on ? 'block' : 'none';
  document.getElementById('loader-text').textContent = msg;
  document.getElementById('search-btn').disabled = on;
  document.getElementById('upload-btn').disabled = on;
}

function showError(msg) {
  const t = document.getElementById('error-toast');
  t.textContent = '⚠️ ' + msg;
  t.style.display = 'block';
}

function hideDashboard() {
  document.getElementById('dashboard').classList.remove('visible');
  document.getElementById('error-toast').style.display = 'none';
}

function showResults(data) {
  document.getElementById('original-img').src = data.satellite_image || data.original_image || '';
  document.getElementById('mask-img').src = data.mask_image || '';
  document.getElementById('stat-coverage').textContent =
    (typeof data.coverage === 'number' ? data.coverage.toFixed(1) + '%' : '--%');

  const p = data.prediction || { emoji: '☀️', label: 'Analysis Result', description: 'Analysis complete.' };
  document.getElementById('pred-emoji').textContent = p.emoji || '☀️';
  document.getElementById('pred-label').textContent = p.label || 'Analysis Result';

  const roofArea = (data.metrics && typeof data.metrics.roof_area_m2 === 'number')
    ? data.metrics.roof_area_m2
    : (typeof data.roof_area_m2 === 'number' ? data.roof_area_m2 : null);

  if (roofArea === null) {
    document.getElementById('pred-desc').textContent =
      (p.description || 'Analysis complete.') + ` Assumption: average usage ${ASSUMED_MONTHLY_KWH_MIN}-${ASSUMED_MONTHLY_KWH_MAX} kWh/month.`;
  } else {
    const m = computeMetricsFromRoofArea(roofArea);

    document.getElementById('val-area').textContent = m.roof_area_m2;
    document.getElementById('val-panels').textContent = m.panel_count;
    document.getElementById('val-capacity').textContent = m.capacity_kw;
    document.getElementById('val-savings').textContent = '$' + m.annual_savings.toLocaleString();
    document.getElementById('val-co2').textContent = m.co2_offset_kg.toLocaleString();

    document.getElementById('pred-desc').textContent =
      `${p.description || 'Analysis complete.'} ` +
      `Assumption: average usage ${ASSUMED_MONTHLY_KWH_MIN}-${ASSUMED_MONTHLY_KWH_MAX} kWh/month (≈ ${m.assumed_monthly_kwh_avg} kWh/month). ` +
      `Estimated annual offset: ${m.offset_kwh.toLocaleString()} kWh → savings shown.`;
  }

  document.getElementById('dashboard').classList.add('visible');

  // Automatically scroll to results if not fully revealed
  const transition = document.getElementById('analyze');
  const rect = transition.getBoundingClientRect();
  if (rect.top > 0) {
    window.scrollTo({
      top: transition.offsetTop + (window.innerHeight * 0.5),
      behavior: 'smooth'
    });
  }
}

async function searchAddress() {
  const addr = document.getElementById('address-input').value.trim();
  if (!addr) { showError('Please enter an address.'); return; }

  hideDashboard();
  setLoading(true, 'Geocoding via Mapbox...');
  document.getElementById('place-details').style.display = 'none';

  try {
    const geoResp = await fetch(`${API}/geocode?address=${encodeURIComponent(addr)}`);
    const geo = await geoResp.json();
    if (!geoResp.ok) throw new Error(geo.detail || 'Geocoding failed');

    const placeDetails = document.getElementById('place-details');
    placeDetails.textContent = '📍 ' + geo.place_name;
    placeDetails.style.display = 'block';

    setLoading(true, 'Running AI segmentation...');
    const segResp = await fetch(`${API}/segment-from-coords?lat=${geo.lat}&lng=${geo.lng}`);
    const seg = await segResp.json();
    if (!segResp.ok) throw new Error(seg.detail || 'Segmentation failed');

    document.getElementById('stat-source').textContent = 'Mapbox';
    showResults(seg);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

let selectedFile = null;
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) pickFile(e.target.files[0]);
});

function pickFile(file) {
  selectedFile = file;
  document.getElementById('preview-name').textContent = '📎 ' + file.name;
  document.getElementById('preview-name').style.display = 'block';
  document.getElementById('upload-btn').style.display = 'block';
  hideDashboard();
}

async function analyzeUpload() {
  if (!selectedFile) return;
  hideDashboard();
  setLoading(true, 'Analyzing rooftop...');

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    const resp = await fetch(`${API}/segment`, { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Analysis failed');

    document.getElementById('stat-source').textContent = 'Local Upload';
    data.satellite_image = data.original_image;
    showResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}
