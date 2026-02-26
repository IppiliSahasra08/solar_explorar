document.addEventListener('DOMContentLoaded', () => {
    const addressInput = document.getElementById('addressInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultsSection = document.getElementById('results');
    const loader = document.getElementById('loader');

    // UI Elements
    const satelliteImg = document.getElementById('satelliteImg');
    const maskImg = document.getElementById('maskImg');
    const roofPercentageText = document.getElementById('roofPercentage');
    const roofProgress = document.getElementById('roofProgress');
    const estAreaText = document.getElementById('estArea');
    const suitabilityText = document.getElementById('suitability');

    const monthlySavingsText = document.getElementById('monthlySavings');
    const annualSavingsText = document.getElementById('annualSavings');
    const paybackText = document.getElementById('payback');

    searchBtn.addEventListener('click', analyzeAddress);
    addressInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeAddress();
    });

    async function analyzeAddress() {
        const address = addressInput.value.trim();
        if (!address) {
            alert('Please enter an address');
            return;
        }

        // Show loading state
        loader.style.display = 'block';
        resultsSection.style.display = 'none';
        searchBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ address })
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update UI with results
            satelliteImg.src = `data:image/jpeg;base64,${data.image_base64}`;
            maskImg.src = `data:image/png;base64,${data.mask_base64}`;
            
            const perc = data.roof_percentage;
            roofPercentageText.textContent = `${perc}%`;
            roofProgress.style.width = `${perc}%`;

            // Calculate estimated metrics (simplified for prototype)
            // Assuming 640x640 image covers roughly 40m x 40m area at zoom 20
            // Total area = 1600 sqm
            const estimatedArea = (perc / 100) * 1600;
            estAreaText.textContent = `${Math.round(estimatedArea)} m²`;

            // Suitability logic
            if (perc > 20) {
                suitabilityText.textContent = 'Excellent';
                suitabilityText.style.color = '#4ade80';
            } else if (perc > 5) {
                suitabilityText.textContent = 'Good';
                suitabilityText.style.color = '#fbbf24';
            } else {
                suitabilityText.textContent = 'Low';
                suitabilityText.style.color = '#f87171';
            }

            // Solar Savings calculation (Dummy logic for now)
            const kwhPerSqm = 150; // annual kwh per sqm of roof
            const pricePerKwh = 0.15;
            const annualGen = estimatedArea * kwhPerSqm;
            const annualSavings = annualGen * pricePerKwh;
            
            monthlySavingsText.textContent = `$${(annualSavings / 12).toFixed(2)}`;
            annualSavingsText.textContent = `$${annualSavings.toFixed(2)}`;
            paybackText.textContent = `${(5000 / (annualSavings / 10)).toFixed(1)} years`;

            // Show results
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            loader.style.display = 'none';
            searchBtn.disabled = false;
        }
    }
});
