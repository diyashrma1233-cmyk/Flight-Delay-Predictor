/**
 * Flight Delay Predictor — Frontend Logic
 * =========================================
 * Handles slider interactions, debounced API calls, and dynamic output updates.
 */

(function () {
    "use strict";

    // ── Feature definitions (id, display precision) ──
    const FEATURES = [
        { id: "departure_hour",         precision: 0 },
        { id: "scheduled_duration_min", precision: 0 },
        { id: "distance_miles",         precision: 0 },
        { id: "num_connections",        precision: 0 },
        { id: "wind_speed_kmh",         precision: 0 },
        { id: "visibility_km",          precision: 0 },
        { id: "prev_flight_delay_min",  precision: 0 },
    ];

    // ── DOM References ──
    const statusWrapper = document.getElementById("status-wrapper");
    const statusText    = document.getElementById("status-text");
    const probBar       = document.getElementById("probability-bar");
    const probValue     = document.getElementById("probability-value");
    const outputCard    = document.getElementById("output-card");

    // ── Debounce utility ──
    function debounce(fn, delay) {
        let timer = null;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    }

    // ── Update slider track fill (CSS gradient) ──
    function updateSliderFill(slider) {
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const val = parseFloat(slider.value);
        const pct = ((val - min) / (max - min)) * 100;
        slider.style.background =
            `linear-gradient(90deg, #6c63ff 0%, #a78bfa ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
    }

    // ── Gather current slider values ──
    function gatherInputs() {
        const payload = {};
        FEATURES.forEach(({ id }) => {
            payload[id] = parseFloat(document.getElementById(id).value);
        });
        return payload;
    }

    // ── Call /predict API ──
    async function fetchPrediction() {
        const payload = gatherInputs();

        // Show loading state
        outputCard.classList.add("loading");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) throw new Error(`Server error ${response.status}`);

            const result = await response.json();
            renderResult(result);
        } catch (err) {
            console.error("Prediction failed:", err);
            statusText.textContent = "Predicted Status: Error";
            statusWrapper.className = "output__status-wrapper";
            probValue.textContent = "—";
            probBar.style.width = "0%";
        } finally {
            outputCard.classList.remove("loading");
        }
    }

    // ── Render prediction result ──
    function renderResult({ prediction, probability }) {
        const isDelayed = prediction === 1;

        // Status text
        statusText.textContent = isDelayed
            ? "Predicted Status: DELAYED"
            : "Predicted Status: ON TIME";

        // Status wrapper class (drives colour via CSS)
        statusWrapper.className = "output__status-wrapper " + (isDelayed ? "delayed" : "on-time");

        // Probability bar + value
        const pctWidth = Math.round(probability * 100);
        probBar.style.width = pctWidth + "%";
        probValue.textContent = probability.toFixed(2);
    }

    // ── Debounced predict ──
    const debouncedPredict = debounce(fetchPrediction, 200);

    // ── Initialise sliders ──
    FEATURES.forEach(({ id, precision }) => {
        const slider  = document.getElementById(id);
        const display = document.getElementById(id + "_value");

        if (!slider || !display) return;

        // Set initial display value & track fill
        display.textContent = parseFloat(slider.value).toFixed(precision);
        updateSliderFill(slider);

        // Listen for input changes
        slider.addEventListener("input", () => {
            display.textContent = parseFloat(slider.value).toFixed(precision);
            updateSliderFill(slider);
            debouncedPredict();
        });
    });

    // ── Initial prediction on page load ──
    fetchPrediction();
})();
