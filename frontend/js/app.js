// ============================================================
// INTELLIGENT MEDIA PROCESSING PIPELINE
// FRONTEND APPLICATION
// ============================================================

const API_BASE_URL = "http://localhost:8000";

// DOM elements
const uploadBox = document.getElementById("uploadBox");
const imageInput = document.getElementById("imageInput");
const browseButton = document.getElementById("browseButton");
const analyzeButton = document.getElementById("analyzeButton");

const previewCard = document.getElementById("previewCard");
const imagePreview = document.getElementById("imagePreview");
const previewTitle = document.getElementById("previewTitle");
const fileName = document.getElementById("fileName");
const fileDetails = document.getElementById("fileDetails");
const removeImageButton = document.getElementById("removeImageButton");

const processingSection = document.getElementById("processingSection");
const processingMessage = document.getElementById("processingMessage");
const progressPercentage = document.getElementById("progressPercentage");
const progressLabel = document.getElementById("progressLabel");
const processingIdDisplay = document.getElementById("processingIdDisplay");
const estimatedTime = document.getElementById("estimatedTime");
const pipelineSteps = document.querySelectorAll(".pipeline-step");
const progressValue = document.querySelector(".progress-value");

const resultsSection = document.getElementById("resultsSection");
const vehicleNumber = document.getElementById("vehicleNumber");
const vehicleMessage = document.getElementById("vehicleMessage");
const vehicleConfidence = document.getElementById("vehicleConfidence");
const vehicleConfidenceBar = document.getElementById("vehicleConfidenceBar");

const overallStatus = document.getElementById("overallStatus");
const overallMessage = document.getElementById("overallMessage");
const overallResultBanner = document.getElementById("overallResultBanner");

const checksGrid = document.getElementById("checksGrid");
const checksSummary = document.getElementById("checksSummary");

const ocrEngine = document.getElementById("ocrEngine");
const ocrConfidence = document.getElementById("ocrConfidence");
const ocrText = document.getElementById("ocrText");

const newAnalysisButton = document.getElementById("newAnalysisButton");
const newAnalysisButtonBottom =
    document.getElementById("newAnalysisButtonBottom");

// Optional analyzed-image elements, if present in index.html
const resultImagePreview = document.getElementById("resultImagePreview");
const resultImageName = document.getElementById("resultImageName");
const resultImageDimensions =
    document.getElementById("resultImageDimensions");
const resultImageSize = document.getElementById("resultImageSize");

// Application state
let selectedFile = null;
let currentProcessingId = null;
let processingTimer = null;
let previewObjectUrl = null;


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    resetProcessingUI();
    hideElement(resultsSection);
    hideElement(processingSection);
});


// ============================================================
// FILE SELECTION
// ============================================================

if (browseButton && imageInput) {
    browseButton.addEventListener("click", (event) => {
        event.stopPropagation();
        imageInput.click();
    });

    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0];
        if (file) handleFile(file);
    });
}

if (uploadBox && imageInput) {
    uploadBox.addEventListener("click", (event) => {
        if (
            event.target !== browseButton &&
            !browseButton?.contains(event.target)
        ) {
            imageInput.click();
        }
    });

    uploadBox.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadBox.classList.add("drag-over");
    });

    uploadBox.addEventListener("dragleave", () => {
        uploadBox.classList.remove("drag-over");
    });

    uploadBox.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadBox.classList.remove("drag-over");

        const file = event.dataTransfer?.files?.[0];
        if (file) handleFile(file);
    });
}

if (removeImageButton) {
    removeImageButton.addEventListener("click", removeSelectedImage);
}

if (analyzeButton) {
    analyzeButton.addEventListener("click", startAnalysis);
}

if (newAnalysisButton) {
    newAnalysisButton.addEventListener("click", resetForNewAnalysis);
}

if (newAnalysisButtonBottom) {
    newAnalysisButtonBottom.addEventListener(
        "click",
        resetForNewAnalysis
    );
}


// ============================================================
// HANDLE SELECTED IMAGE
// ============================================================


function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Image size must be 10 MB or less.");
        return;
    }

    selectedFile = file;

    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
    }

    previewObjectUrl = URL.createObjectURL(file);

    if (imagePreview) {
        imagePreview.onload = function () {
            if (fileDetails) {
                fileDetails.textContent =
                    `${formatFileSize(file.size)} â€¢ ` +
                    `${imagePreview.naturalWidth} Ã— ${imagePreview.naturalHeight}`;
            }
        };

        imagePreview.onerror = function () {
            console.error("Failed to display selected image.");
            alert("Unable to preview this image. Please choose another.");
        };

        imagePreview.src = previewObjectUrl;
        imagePreview.alt = file.name;
        imagePreview.style.display = "block";
    }

    if (fileName) {
        fileName.textContent = file.name;
    }

    if (previewTitle) {
        previewTitle.textContent = file.name;
    }

    if (fileDetails && (!imagePreview || !imagePreview.complete)) {
        fileDetails.textContent = formatFileSize(file.size);
    }

    showElement(previewCard);
    hideElement(resultsSection);
    hideElement(processingSection);

    if (analyzeButton) {
        analyzeButton.disabled = false;
    }
}



// ============================================================
// REMOVE SELECTED IMAGE
// ============================================================

function removeSelectedImage() {
    selectedFile = null;

    if (imageInput) {
        imageInput.value = "";
    }

    if (imagePreview) {
        imagePreview.removeAttribute("src");
    }

    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }

    hideElement(previewCard);

    if (analyzeButton) {
        analyzeButton.disabled = true;
    }
}


// ============================================================
// START ANALYSIS
// ============================================================

async function startAnalysis() {
    if (!selectedFile) {
        alert("Please select an image first.");
        return;
    }

    hideElement(resultsSection);
    showElement(processingSection);

    if (analyzeButton) {
        analyzeButton.disabled = true;
    }

    resetProcessingUI();

    try {
        setProcessingProgress(5, "Preparing image...");
        setProcessingStage(0, "active");

        // Upload image to the backend
        setProcessingProgress(10, "Uploading image...");
        setProcessingStage(0, "completed");

        const processingId = await uploadImage();

        currentProcessingId = processingId;

        if (processingIdDisplay) {
            processingIdDisplay.textContent = processingId;
        }

        // Start the visual progress animation
        startProcessingAnimation();

        // Wait until the backend finishes processing
        await waitForProcessing(processingId);

        stopProcessingAnimation();
        completeAllProcessingStages();

        setProcessingProgress(100, "Analysis complete");

        // Retrieve the final analysis result
        const results = await fetchResults(processingId);

        displayResults(results);

        hideElement(processingSection);
        showElement(resultsSection);

        if (resultsSection) {
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    } catch (error) {
        stopProcessingAnimation();

        console.error("Analysis error:", error);

        showError(error.message || "Unable to analyze this image.");
    } finally {
        if (analyzeButton) {
            analyzeButton.disabled = !selectedFile;
        }
    }
}


// ============================================================
// UPLOAD IMAGE
// ============================================================

async function uploadImage() {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(
        `${API_BASE_URL}/api/v1/uploads`,
        {
            method: "POST",
            body: formData
        }
    );

    if (!response.ok) {
        const message = await response.text();
        throw new Error(`Upload failed: ${message}`);
    }

    const data = await response.json();

    console.log("Upload response:", data);

    const processingId = data.processing_id || data.id;

    if (!processingId) {
        throw new Error(
            "The backend did not return a processing ID."
        );
    }

    return processingId;
}


// ============================================================
// POLL PROCESSING STATUS
// ============================================================


async function waitForProcessing(processingId) {
    const maxAttempts = 180;
    const intervalMs = 2000;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const response = await fetch(
            `${API_BASE_URL}/api/v1/processing/${encodeURIComponent(processingId)}`
        );

        if (!response.ok) {
            throw new Error(
                `Status request failed: HTTP ${response.status}`
            );
        }

        const data = await response.json();
        console.log("Processing status:", data);

        const status = String(
            data.status ||
            data.processing_status ||
            data.state ||
            ""
        ).trim().toLowerCase();

        if (
            ["completed", "complete", "success", "succeeded", "finished"]
                .includes(status)
        ) {
            return data;
        }

        if (["failed", "failure", "error"].includes(status)) {
            throw new Error(
                data.failure_reason ||
                data.message ||
                data.error ||
                "Image processing failed."
            );
        }

        if (processingMessage) {
            processingMessage.textContent =
                status === "queued" || status === "pending"
                    ? "Your image is waiting in the queue..."
                    : "AI is analyzing your image...";
        }

        await sleep(intervalMs);
    }

    throw new Error(
        "Processing timed out. Check the backend processing status."
    );
}


function getProcessingMessage(status) {
    switch (status) {
        case "pending":
            return "Image is waiting in the processing queue...";
        case "queued":
            return "Image queued for analysis...";
        case "processing":
            return "AI is analyzing the image...";
        default:
            return "Processing image...";
    }
}


// ============================================================
// FETCH RESULTS
// ============================================================

async function fetchResults(processingId) {
    if (processingMessage) {
        processingMessage.textContent = "Loading analysis results...";
    }

    const response = await fetch(
        `${API_BASE_URL}/api/v1/processing/${encodeURIComponent(processingId)}/results`
    );

    if (!response.ok) {
        const message = await response.text();
        throw new Error(`Unable to retrieve results: ${message}`);
    }

    return await response.json();
}
// ============================================================
// DISPLAY ANALYSIS RESULTS
// ============================================================

function displayResults(data) {
    console.log("Final analysis results:", data);

    const checks = data.checks || {};
    const vehicle = checks.vehicle_number || {};
    const ocr = checks.ocr || {};


    // Registration state and RTO region
    const location = checks.registration_location || {};

    const locationFields = {
        registrationState: location.registration_state,
        registrationStateCode: location.registration_state_code,
        registrationRtoCode: location.rto_code,
        registrationRegion: location.registration_region,
        registrationDistrict: location.registration_district,
        registrationSeries: location.registration_series
    };

    for (const [id, value] of Object.entries(locationFields)) {
        const element = document.getElementById(id);

        if (element) {
            element.textContent = value || "Not available";
        }
    }

    const locationMessage = document.getElementById(
        "registrationLocationMessage"
    );

    if (locationMessage) {
        locationMessage.textContent =
            location.message || "Registration location was not identified.";
    }


    // Vehicle number
    const detectedPlate =
        vehicle.normalized_candidate ||
        vehicle.detected_number ||
        vehicle.plate_number ||
        vehicle.text ||
        "Not detected";

    if (vehicleNumber) {
        vehicleNumber.textContent = detectedPlate;
    }

    if (vehicleMessage) {
        vehicleMessage.textContent =
            vehicle.message ||
            (detectedPlate === "Not detected"
                ? "No valid vehicle number detected."
                : "Vehicle number recognition completed.");
    }

    // Vehicle-number confidence
    const confidence = getConfidence(
        vehicle.confidence ??
        ocr.confidence ??
        0
    );

    if (vehicleConfidence) {
        vehicleConfidence.textContent =
            `${Math.round(confidence)}%`;
    }

    if (vehicleConfidenceBar) {
        vehicleConfidenceBar.style.width =
            `${Math.min(100, Math.max(0, confidence))}%`;
    }

    // OCR details
    if (ocrEngine) {
        ocrEngine.textContent =
            ocr.engine ||
            ocr.ocr_engine ||
            ocr.method ||
            "Not available";
    }

    if (ocrConfidence) {
        const ocrScore = getConfidence(ocr.confidence ?? 0);
        ocrConfidence.textContent = `${Math.round(ocrScore)}%`;
    }

    if (ocrText) {
        ocrText.textContent =
            ocr.text ||
            ocr.raw_text ||
            ocr.detected_text ||
            vehicle.raw_candidate ||
            vehicle.normalized_candidate ||
            "No text detected";
    }

    // Analyzed image details, if the elements exist
    if (resultImagePreview && previewObjectUrl) {
        resultImagePreview.src = previewObjectUrl;
    }

    if (resultImageName && selectedFile) {
        resultImageName.textContent = selectedFile.name;
    }

    if (resultImageSize && selectedFile) {
        resultImageSize.textContent =
            formatFileSize(selectedFile.size);
    }

    if (resultImageDimensions && selectedFile) {
        getImageDimensions(selectedFile).then((dimensions) => {
            resultImageDimensions.textContent = dimensions;
        });
    }

    // Inspection checks
    renderChecks(checks);

    // Overall result
    updateOverallBanner(data, checks, vehicle);
}


// ============================================================
// RENDER INSPECTION CHECKS
// ============================================================

function renderChecks(checks) {
    if (!checksGrid) return;

    checksGrid.innerHTML = "";

    const definitions = [
        { key: "blur", label: "Blur" },
        { key: "brightness", label: "Brightness" },
        { key: "dimensions", label: "Dimensions" },
        { key: "metadata", label: "Metadata" },
        { key: "duplicate", label: "Duplicate" },
        { key: "tampering", label: "Tampering" },
        { key: "screenshot", label: "Screenshot" },
        { key: "photo_of_photo", label: "Photo of Photo" }
    ];

    let passed = 0;
    let warnings = 0;
    let failed = 0;
    let displayed = 0;

    definitions.forEach((definition) => {
        const check = checks[definition.key];

        // Don't create empty cards for missing checks
        if (check === undefined || check === null) return;

        const status = normalizeCheckStatus(check);
        const message = getCheckMessage(check);
        const title = definition.label;

        displayed++;

        if (status === "pass") passed++;
        else if (status === "fail") failed++;
        else warnings++;

        const card = document.createElement("div");
        card.className = `check-card check-${status}`;

        const icon = document.createElement("div");
        icon.className = "check-icon";
        icon.textContent = getStatusIcon(status);

        const content = document.createElement("div");
        content.className = "check-content";

        const heading = document.createElement("h4");
        heading.textContent = title;

        const description = document.createElement("p");
        description.textContent = message;

        const badge = document.createElement("span");
        badge.className = `check-status status-${status}`;
        badge.textContent = formatStatus(status);

        content.appendChild(heading);
        content.appendChild(description);
        content.appendChild(badge);

        card.appendChild(icon);
        card.appendChild(content);

        checksGrid.appendChild(card);
    });

    if (checksSummary) {
        checksSummary.textContent =
            `${passed} passed â€¢ ${warnings} warnings â€¢ ${failed} failed`;
    }

    if (displayed === 0) {
        const emptyMessage = document.createElement("p");
        emptyMessage.textContent =
            "No inspection checks were returned by the backend.";
        checksGrid.appendChild(emptyMessage);
    }
}


// ============================================================
// NORMALIZE CHECK STATUS
// ============================================================

function normalizeCheckStatus(check) {
    if (typeof check === "boolean") {
        return check ? "pass" : "fail";
    }

    if (typeof check === "string") {
        const value = check.toLowerCase();

        if (
            ["pass", "passed", "ok", "clear", "success", "good", "true"]
                .includes(value)
        ) {
            return "pass";
        }

        if (
            ["fail", "failed", "error", "invalid", "false", "rejected"]
                .includes(value)
        ) {
            return "fail";
        }

        return "warning";
    }

    if (!check || typeof check !== "object") {
        return "warning";
    }

    const rawStatus = String(
        check.status ||
        check.result ||
        check.verdict ||
        ""
    ).toLowerCase();

    if (
        ["pass", "passed", "ok", "clear", "success", "good"]
            .includes(rawStatus)
    ) {
        return "pass";
    }

    if (
        ["fail", "failed", "error", "invalid", "rejected"]
            .includes(rawStatus)
    ) {
        return "fail";
    }

    if (
        ["warning", "warn", "suspicious", "unknown", "inconclusive"]
            .includes(rawStatus)
    ) {
        return "warning";
    }

    // Interpret common boolean result fields
    if (typeof check.passed === "boolean") {
        return check.passed ? "pass" : "fail";
    }

    if (typeof check.is_valid === "boolean") {
        return check.is_valid ? "pass" : "fail";
    }

    if (typeof check.detected === "boolean") {
        // Detection can mean a problem was found.
        // For these checks, detected often means a warning.
        return check.detected ? "warning" : "pass";
    }

    return "warning";
}


// ============================================================
// CHECK MESSAGES AND LABELS
// ============================================================

function getCheckMessage(check) {
    if (typeof check === "string") return check;

    if (typeof check === "boolean") {
        return check ? "Check passed." : "Check failed.";
    }

    if (!check || typeof check !== "object") {
        return "No details available.";
    }

    return (
        check.message ||
        check.details ||
        check.description ||
        check.reason ||
        check.status ||
        "Check completed."
    );
}

function getStatusIcon(status) {
    if (status === "pass") return "âœ“";
    if (status === "fail") return "âœ•";
    return "!";
}

function formatStatus(status) {
    if (status === "pass") return "Passed";
    if (status === "fail") return "Failed";
    return "Warning";
}


// ============================================================
// OVERALL RESULT BANNER
// ============================================================


function updateOverallBanner(data, checks, vehicle) {
    const summary = data.summary || {};

    const backendStatus = String(
        summary.overall_status ||
        data.overall_status ||
        data.status ||
        ""
    ).trim().toLowerCase();

    const statuses = Object.keys(checks)
        .filter(
            key => key !== "ocr" && key !== "vehicle_number"
        )
        .map(key => normalizeCheckStatus(checks[key]));

    const hasFailedChecks = statuses.includes("fail");
    const hasWarnings = statuses.includes("warning");

    let finalStatus;
    let message;
    let bannerClass;
    let icon;

    // Use the backend's explicit overall decision first.
    if (
        ["rejected", "failed", "fail", "invalid"].includes(backendStatus)
    ) {
        finalStatus = "Rejected";
        message =
            summary.message ||
            "The image did not meet the required validation criteria.";
        bannerClass = "rejected";
        icon = "âœ•";
    } else if (
        ["approved", "accepted", "passed", "success", "valid"].includes(
            backendStatus
        )
    ) {
        finalStatus = "Approved";
        message =
            summary.message ||
            "The image passed the overall validation.";
        bannerClass = "approved";
        icon = "âœ“";
    } else if (hasFailedChecks) {
        finalStatus = "Needs Review";
        message = "One or more image-quality checks failed.";
        bannerClass = "warning";
        icon = "!";
    } else if (hasWarnings) {
        finalStatus = "Needs Review";
        message = "Some checks require further review.";
        bannerClass = "warning";
        icon = "!";
    } else {
        finalStatus = "Processed";
        message = "Image analysis has completed.";
        bannerClass = "success";
        icon = "|";
    }

    if (overallStatus) {
        overallStatus.textContent = finalStatus;
    }

    if (overallMessage) {
        overallMessage.textContent = message;
    }

    if (overallResultBanner) {
        // Remove old visual states before applying the new one.
        overallResultBanner.classList.remove(
            "approved",
            "rejected",
            "warning",
            "success"
        );

        overallResultBanner.classList.add(bannerClass);

        // Update the banner icon if the HTML has an icon element.
        const iconElement = overallResultBanner.querySelector(
            ".result-icon, .status-icon, .overall-icon"
        );

        if (iconElement) {
            iconElement.textContent = icon;
        }
    }
}

// ============================================================
// PROCESSING UI
// ============================================================

function resetProcessingUI() {
    stopProcessingAnimation();

    pipelineSteps.forEach((step) => {
        step.classList.remove(
            "active",
            "completed",
            "done",
            "failed"
        );
    });

    setProcessingProgress(0, "Waiting to start...");

    if (processingIdDisplay) {
        processingIdDisplay.textContent = "Generating...";
    }

    if (estimatedTime) {
        estimatedTime.textContent = "Please wait...";
    }
}

function setProcessingStage(index, status) {
    const step = document.querySelector(
        `.pipeline-step[data-stage="${index}"]`
    );

    if (!step) return;

    step.classList.remove(
        "active",
        "completed",
        "done",
        "failed"
    );

    if (status === "completed") {
        step.classList.add("completed");
    } else if (status === "active") {
        step.classList.add("active");
    } else if (status === "failed") {
        step.classList.add("failed");
    }
}

function setProcessingProgress(percent, message) {
    const value = Math.max(0, Math.min(100, percent));

    if (progressPercentage) {
        progressPercentage.textContent = `${Math.round(value)}%`;
    }

    if (progressLabel) {
        progressLabel.textContent = message || "Processing...";
    }

    if (processingMessage && message) {
        processingMessage.textContent = message;
    }

    if (progressValue) {
        progressValue.style.strokeDasharray = `${value}, 100`;
    }
}

function startProcessingAnimation() {
    stopProcessingAnimation();

    let progress = 10;
    let stage = 1;

    processingTimer = setInterval(() => {
        // This is only a visual indicator.
        // The backend polling determines when processing finishes.
        if (progress < 90) {
            progress += Math.random() * 3;
            progress = Math.min(progress, 90);

            setProcessingProgress(
                progress,
                getStageMessage(stage)
            );

            if (stage < pipelineSteps.length) {
                setProcessingStage(stage, "active");
            }

            if (progress > stage * 15 && stage < pipelineSteps.length) {
                setProcessingStage(stage, "completed");
                stage++;
            }
        }
    }, 1000);
}

function stopProcessingAnimation() {
    if (processingTimer) {
        clearInterval(processingTimer);
        processingTimer = null;
    }
}

function getStageMessage(stage) {
    const messages = [
        "Uploading image...",
        "Checking image quality...",
        "Analyzing brightness and sharpness...",
        "Checking image authenticity...",
        "Recognizing vehicle number...",
        "Preparing results..."
    ];

    return messages[Math.min(stage, messages.length - 1)];
}

function completeAllProcessingStages() {
    stopProcessingAnimation();

    pipelineSteps.forEach((step) => {
        step.classList.remove("active", "failed");
        step.classList.add("completed");
    });
}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {
    hideElement(processingSection);

    if (analyzeButton) {
        analyzeButton.disabled = !selectedFile;
    }

    alert(message);
}


// ============================================================
// RESET FOR A NEW ANALYSIS
// ============================================================

function resetForNewAnalysis() {
    stopProcessingAnimation();

    currentProcessingId = null;

    hideElement(resultsSection);
    hideElement(processingSection);

    resetProcessingUI();

    if (selectedFile) {
        showElement(previewCard);
    } else {
        hideElement(previewCard);
    }

    if (analyzeButton) {
        analyzeButton.disabled = !selectedFile;
    }

    if (selectedFile && uploadBox) {
        uploadBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    } else if (uploadBox) {
        uploadBox.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
}


// ============================================================
// HELPER FUNCTIONS
// ============================================================


function showElement(element) {
    if (!element) return;

    element.classList.remove("hidden");
    element.hidden = false;
    element.style.display = "";
}

function hideElement(element) {
    if (!element) return;

    element.classList.add("hidden");
    element.hidden = true;
}


function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) {
        return "Unknown size";
    }

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getConfidence(value) {
    let confidence = Number(value);

    if (!Number.isFinite(confidence)) {
        return 0;
    }

    // Convert a fractional confidence such as 0.92 to 92%.
    if (confidence > 0 && confidence <= 1) {
        confidence *= 100;
    }

    return Math.max(0, Math.min(100, confidence));
}

function getImageDimensions(file) {
    return new Promise((resolve) => {
        const image = new Image();
        const url = URL.createObjectURL(file);

        image.onload = () => {
            resolve(`${image.naturalWidth} Ã— ${image.naturalHeight}`);
            URL.revokeObjectURL(url);
        };

        image.onerror = () => {
            resolve("Unavailable");
            URL.revokeObjectURL(url);
        };

        image.src = url;
    });
}

function sleep(milliseconds) {
    return new Promise((resolve) => {
        setTimeout(resolve, milliseconds);
    });
}


const historyList = document.getElementById("historyList");
const historySearch = document.getElementById("historySearch");
const historyMessage = document.getElementById("historyMessage");
const refreshHistoryButton = document.getElementById("refreshHistoryButton");
const searchHistoryButton = document.getElementById("searchHistoryButton");

async function loadAnalysisHistory(search = "") {
    if (!historyList) return;

    historyList.replaceChildren();
    historyList.textContent = "Loading history...";

    if (historyMessage) historyMessage.textContent = "";

    try {
        const params = new URLSearchParams({ limit: "50" });
        if (search.trim()) params.set("search", search.trim());

        const response = await fetch(
            `${API_BASE_URL}/api/v1/processing/history?${params}`
        );

        if (!response.ok) {
            throw new Error(`History request failed: HTTP ${response.status}`);
        }

        const data = await response.json();
        const items = Array.isArray(data.items) ? data.items : [];

        historyList.replaceChildren();

        if (items.length === 0) {
            historyList.textContent = "No analysis history found.";
            return;
        }

        items.forEach((item) => {
            const card = document.createElement("article");
            card.className = "history-card";

            const heading = document.createElement("h3");
            heading.textContent = item.filename || "Unnamed image";

            const id = document.createElement("p");
            id.textContent = `Processing ID: ${item.processing_id || "Unavailable"}`;


            const date = document.createElement("p");

            date.textContent = `Uploaded: ${
                item.created_at
                    ? new Date(
                            item.created_at.endsWith("Z")
                            ? item.created_at
                            : item.created_at + "Z"
                    ).toLocaleString("en-IN", {
                        timeZone: "Asia/Kolkata",
                        day: "2-digit",
                        month: "2-digit",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                        hour12: true
                    })
                : "Unknown"
            }`;


            const status = document.createElement("p");
            status.textContent = `Processing status: ${item.status || "Unknown"}`;

            const verdict = document.createElement("p");
            verdict.textContent = `Overall result: ${
                formatHistoryVerdict(item.overall_status)
            }`;

            const vehicle = document.createElement("p");
            vehicle.textContent = `Vehicle number: ${item.vehicle_number || "Not detected"}`;

            const actions = document.createElement("div");
            actions.className = "history-actions";

            const copyButton = document.createElement("button");
            copyButton.type = "button";
            copyButton.className = "secondary-button";
            copyButton.textContent = "Copy ID";
            copyButton.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(item.processing_id);
                    if (historyMessage) historyMessage.textContent = "Processing ID copied.";
                } catch {
                    if (historyMessage) historyMessage.textContent =
                        "Unable to copy automatically. Select and copy the ID manually.";
                }
            });

            const viewButton = document.createElement("button");
            viewButton.type = "button";
            viewButton.className = "primary-button";
            viewButton.textContent = "View Results";
            viewButton.disabled = item.status !== "completed";
            viewButton.addEventListener("click", () => {
                viewHistoryResults(item.processing_id);
            });

            actions.append(copyButton, viewButton);
            card.append(heading, id, date, status, verdict, vehicle, actions);
            historyList.appendChild(card);
        });

        if (historyMessage) {
            historyMessage.textContent = `${items.length} record(s) found.`;
        }
    } catch (error) {
        console.error("Unable to load analysis history:", error);
        historyList.textContent =
            "Could not load history. Check that the backend is running and try refreshing.";
        if (historyMessage) historyMessage.textContent = error.message;
    }
}

function formatHistoryVerdict(value) {
    const verdicts = {
        accepted: "Accepted",
        approved: "Approved",
        review_recommended: "Review recommended",
        rejected: "Rejected"
    };

    if (!value) return "Not available";
    const key = String(value).trim().toLowerCase();
    return verdicts[key] || String(value).replaceAll("_", " ");
}

async function viewHistoryResults(processingId) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/v1/processing/${encodeURIComponent(processingId)}/results`
        );

        if (!response.ok) {
            throw new Error(`Unable to retrieve results: HTTP ${response.status}`);
        }

        const data = await response.json();

        // Reuse the existing results renderer.
        displayResults(data);

        if (resultsSection) {
            showElement(resultsSection);
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    } catch (error) {
        console.error("Unable to open historical results:", error);
        alert(error.message || "Unable to load these results.");
    }
}

if (refreshHistoryButton) {
    refreshHistoryButton.addEventListener("click", () => {
        loadAnalysisHistory(historySearch?.value || "");
    });
}

if (searchHistoryButton) {
    searchHistoryButton.addEventListener("click", () => {
        loadAnalysisHistory(historySearch?.value || "");
    });
}

if (historySearch) {
    historySearch.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            loadAnalysisHistory(historySearch.value);
        }
    });
}

/*document.addEventListener("DOMContentLoaded", () => {
    loadAnalysisHistory();
});*/

const historyButton = document.getElementById("historyButton");
const historySection = document.getElementById("historySection");

if (historyButton && historySection) {
    historyButton.addEventListener("click", async () => {
        const isHidden = historySection.hidden;

        historySection.hidden = !isHidden;
        historySection.classList.toggle("hidden", !isHidden);

        historyButton.textContent = isHidden
            ? "Hide History"
            : "History";

        if (isHidden) {
            await loadAnalysisHistory(historySearch?.value || "");

            historySection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }
    });
}
