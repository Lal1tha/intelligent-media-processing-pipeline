const API_BASE_URL = "http://localhost:8000";

// =========================================================
// DOM ELEMENTS
// =========================================================

const uploadBox = document.getElementById("uploadBox");

const imageInput = document.getElementById("imageInput");

const browseButton = document.getElementById("browseButton");

const analyzeButton = document.getElementById("analyzeButton");

const fileName = document.getElementById("fileName");

const processingSection =
    document.getElementById("processingSection");

const processingMessage =
    document.getElementById("processingMessage");

const resultsSection =
    document.getElementById("resultsSection");

const vehicleNumber =
    document.getElementById("vehicleNumber");

const vehicleMessage =
    document.getElementById("vehicleMessage");

const vehicleConfidence =
    document.getElementById("vehicleConfidence");

const overallStatus =
    document.getElementById("overallStatus");

const checksGrid =
    document.getElementById("checksGrid");

const ocrEngine =
    document.getElementById("ocrEngine");

const ocrConfidence =
    document.getElementById("ocrConfidence");

const ocrText =
    document.getElementById("ocrText");

const newAnalysisButton =
    document.getElementById("newAnalysisButton");


// =========================================================
// STATE
// =========================================================

let selectedFile = null;


// =========================================================
// FILE SELECTION
// =========================================================

browseButton.addEventListener("click", () => {
    imageInput.click();
});


uploadBox.addEventListener("click", (event) => {

    if (
        event.target !== browseButton &&
        !browseButton.contains(event.target)
    ) {
        imageInput.click();
    }

});


imageInput.addEventListener("change", () => {

    if (imageInput.files.length === 0) {
        return;
    }

    handleFile(imageInput.files[0]);

});


function handleFile(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image file.");

        return;
    }

    selectedFile = file;

    fileName.textContent = file.name;

    analyzeButton.disabled = false;

}


// =========================================================
// DRAG AND DROP
// =========================================================

uploadBox.addEventListener("dragover", (event) => {

    event.preventDefault();

    uploadBox.classList.add("dragover");

});


uploadBox.addEventListener("dragleave", () => {

    uploadBox.classList.remove("dragover");

});


uploadBox.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadBox.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length === 0) {
        return;
    }

    handleFile(files[0]);

});


// =========================================================
// START ANALYSIS
// =========================================================

analyzeButton.addEventListener(
    "click",
    startAnalysis
);


function showProcessing(show = true) {

    const processingSection =
        document.getElementById("processingSection");

    const uploadSection =
        document.getElementById("uploadSection");

    const resultsSection =
        document.getElementById("resultsSection");

    if (processingSection) {

        processingSection.classList.toggle(
            "hidden",
            !show
        );

    }

    if (uploadSection) {

        uploadSection.classList.toggle(
            "hidden",
            show
        );

    }

    if (resultsSection) {

        resultsSection.classList.add("hidden");

    }

}


async function startAnalysis() {

    if (!selectedFile) {

        alert("Please select an image first.");

        return;
    }

    showProcessing();

    try {

        const processingId =
            await uploadImage();

        await waitForProcessing(
            processingId
        );

        const results =
            await fetchResults(processingId);

        displayResults(results);

    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Something went wrong while processing the image."
        );

    }

}


// =========================================================
// UPLOAD IMAGE
// =========================================================

async function uploadImage() {

    const formData = new FormData();

    formData.append(
        "file",
        selectedFile
    );

    processingMessage.textContent =
        "Uploading image...";

    const response = await fetch(
        `${API_BASE_URL}/api/v1/uploads`,
        {
            method: "POST",
            body: formData
        }
    );

    if (!response.ok) {

        const errorText =
            await response.text();

        throw new Error(
            `Upload failed: ${errorText}`
        );

    }

    const data =
        await response.json();

    console.log(
        "Upload response:",
        data
    );

    // Backend should return a processing ID.
    const processingId =
        data.processing_id ||
        data.id;

    if (!processingId) {

        throw new Error(
            "Backend did not return a processing ID."
        );

    }

    return processingId;

}


// =========================================================
// WAIT FOR PROCESSING
// =========================================================

async function waitForProcessing(
    processingId
) {

    // Maximum 180 attempts.
    // Each attempt waits 2 seconds.
    // Maximum waiting time = 6 minutes.
    const maxAttempts = 180;

    const interval = 2000;

    for (
        let attempt = 0;
        attempt < maxAttempts;
        attempt++
    ) {

        processingMessage.textContent =
            `Processing image... (${attempt + 1})`;

        const response = await fetch(
            `${API_BASE_URL}/api/v1/processing/${processingId}`
        );

        if (!response.ok) {

            throw new Error(
                "Unable to check processing status."
            );

        }

        const data =
            await response.json();

        console.log(
            "Processing status:",
            data
        );

        const status =
            (
                data.status ||
                data.processing_status ||
                ""
            ).toLowerCase();


        if (
            status === "completed" ||
            status === "complete" ||
            status === "success"
        ) {

            return;

        }


        if (
            status === "failed" ||
            status === "error"
        ) {

            throw new Error(
                data.message ||
                "Image processing failed."
            );

        }


        await sleep(interval);

    }

    throw new Error(
        "Processing timed out. Please try again."
    );

}


// =========================================================
// FETCH RESULTS
// =========================================================

async function fetchResults(
    processingId
) {

    processingMessage.textContent =
        "Loading analysis results...";

    const response = await fetch(
        `${API_BASE_URL}/api/v1/processing/${processingId}/results`
    );

    if (!response.ok) {

        throw new Error(
            "Unable to retrieve analysis results."
        );

    }

    return await response.json();

}


// =========================================================
// DISPLAY RESULTS
// =========================================================

function displayResults(data) {

    console.log(
        "Final results:",
        data
    );

    const checks =
        data.checks || {};

    const vehicle =
        checks.vehicle_number || {};

    const ocr =
        checks.ocr || {};


    // =====================================================
    // VEHICLE NUMBER
    // =====================================================

    vehicleNumber.textContent =
        vehicle.normalized_candidate ||
        "Not detected";


    vehicleMessage.textContent =
        vehicle.message ||
        "No message available.";


    const confidence =
        Number(vehicle.confidence || 0);


    vehicleConfidence.textContent =
        `${Math.round(confidence * 100)}%`;


    // =====================================================
    // OVERALL STATUS
    // =====================================================

    const summary =
        data.summary || {};


    overallStatus.textContent =
        formatStatus(
            summary.overall_status ||
            data.status ||
            "Review"
        );


    // =====================================================
    // INDIVIDUAL CHECKS
    // =====================================================

    renderChecks(checks);


    // =====================================================
    // OCR / PLATE RECOGNITION
    // =====================================================

    /*
     * IMPORTANT:
     *
     * The backend has two different OCR-related outputs:
     *
     * 1. checks.ocr
     *    → full-image OCR
     *    → may contain advertisements, boards, etc.
     *
     * 2. checks.vehicle_number
     *    → actual vehicle plate detection
     *    → normalized_candidate = MH12KR1145
     *
     * For the UI we want to show the useful
     * vehicle plate result instead of noisy
     * full-image advertisement text.
     */

    const detectedPlate =
        vehicle.normalized_candidate ||
        "Not detected";


    /*
     * Prefer the OCR engines reported by
     * the vehicle-number detector.
     */
    const plateEngines =
        Array.isArray(vehicle.ocr_engines) &&
        vehicle.ocr_engines.length > 0
            ? vehicle.ocr_engines.join(", ")
            : ocr.engine || "—";


    /*
     * Prefer vehicle-number confidence.
     * Fall back to general OCR confidence
     * if necessary.
     */
    const plateConfidence =
        Number(
            vehicle.confidence ||
            ocr.confidence ||
            0
        );


    ocrEngine.textContent =
        plateEngines;


    ocrConfidence.textContent =
        plateConfidence > 0
            ? `${Math.round(plateConfidence * 100)}%`
            : "—";


    /*
     * Show the detected vehicle number,
     * NOT the noisy full-image OCR text.
     */
    ocrText.textContent =
        detectedPlate;


    // =====================================================
    // SHOW RESULTS
    // =====================================================

    processingSection.classList.add(
        "hidden"
    );

    resultsSection.classList.remove(
        "hidden"
    );


    window.scrollTo({
        top: resultsSection.offsetTop - 40,
        behavior: "smooth"
    });

}


// =========================================================
// RENDER CHECK CARDS
// =========================================================

function renderChecks(checks) {

    checksGrid.innerHTML = "";

    const checkDefinitions = [

        {
            key: "blur",
            label: "Blur"
        },

        {
            key: "brightness",
            label: "Brightness"
        },

        {
            key: "dimensions",
            label: "Dimensions"
        },

        {
            key: "metadata",
            label: "Metadata"
        },

        {
            key: "duplicate",
            label: "Duplicate"
        },

        {
            key: "tampering",
            label: "Tampering"
        },

        {
            key: "screenshot",
            label: "Screenshot"
        },

        {
            key: "photo_of_photo",
            label: "Photo of Photo"
        }

    ];


    checkDefinitions.forEach(
        definition => {

            const check =
                checks[definition.key];

            if (!check) {
                return;
            }


            const status =
                normalizeCheckStatus(
                    check.status
                );


            const card =
                document.createElement("div");


            card.className =
                "check-card";


            card.innerHTML = `

                <div class="check-card-header">

                    <div class="check-icon ${status}">
                        ${getStatusIcon(status)}
                    </div>

                    <div>

                        <h4>
                            ${definition.label}
                        </h4>

                        <span class="check-status">
                            ${formatStatus(status)}
                        </span>

                    </div>

                </div>

                <p class="check-message">
                    ${escapeHtml(
                        check.message ||
                        "No additional information."
                    )}
                </p>

            `;


            checksGrid.appendChild(card);

        }
    );

}


// =========================================================
// STATUS HELPERS
// =========================================================

function normalizeCheckStatus(status) {

    const value =
        String(status || "")
            .toLowerCase();


    if (
        value === "pass" ||
        value === "passed" ||
        value === "success"
    ) {

        return "pass";

    }


    if (
        value === "warning" ||
        value === "warn"
    ) {

        return "warning";

    }


    return "fail";

}


function getStatusIcon(status) {

    if (status === "pass") {
        return "✓";
    }


    if (status === "warning") {
        return "!";
    }


    return "×";

}


function formatStatus(status) {

    return String(status || "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, char =>
            char.toUpperCase()
        );

}


// =========================================================
// ERROR
// =========================================================

function showError(message) {

    processingSection.classList.remove(
        "hidden"
    );

    resultsSection.classList.add(
        "hidden"
    );

    processingMessage.textContent =
        message;

}


// =========================================================
// NEW ANALYSIS
// =========================================================

newAnalysisButton.addEventListener(
    "click",
    () => {

        selectedFile = null;

        imageInput.value = "";

        fileName.textContent = "";

        analyzeButton.disabled = true;

        resultsSection.classList.add(
            "hidden"
        );

        processingSection.classList.add(
            "hidden"
        );

        const uploadSection =
            document.getElementById("uploadSection");

        if (uploadSection) {

            uploadSection.classList.remove(
                "hidden"
            );

        }

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);


// =========================================================
// UTILITIES
// =========================================================

function sleep(milliseconds) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );

}


function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent = value;

    return div.innerHTML;

}