const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const progressContainer = document.getElementById('progress-bar-container');
const progressBar = document.getElementById('file-upload-progress');
const predictionResult = document.getElementById('prediction-result');
const predictionText = document.getElementById('prediction-text');
const uploadedImage = document.getElementById('uploaded-image');
const feedbackSection = document.getElementById('feedback-section');
const commentBox = document.getElementById('comment');
const acceptBtn = document.getElementById('accept-btn');
const rejectBtn = document.getElementById('reject-btn');

let currentPrediction = "";
let currentFile = null;

// ✅ UPDATED: Set this to your deployed backend URL
const BACKEND_URL = "https://diabetic-retinopathy-with-feedback.onrender.com";

// Clicking upload area triggers file input
uploadArea.addEventListener('click', () => fileInput.click());

// Optional: Drag and drop support
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.backgroundColor = '#e6f0fa';
});
uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.style.backgroundColor = '';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.backgroundColor = '';
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileUpload();
    }
});

fileInput.addEventListener('change', handleFileUpload);

// Check backend availability
async function isBackendAvailable() {
    try {
        const res = await fetch(`${BACKEND_URL}/`, { method: 'GET' });
        return res.ok;
    } catch {
        return false;
    }
}

async function handleFileUpload() {
    if (fileInput.files.length === 0) return;
    currentFile = fileInput.files[0];

    const formData = new FormData();
    formData.append('file', currentFile);

    progressContainer.style.display = 'block';
    progressBar.value = 0;
    predictionResult.style.display = 'none';
    feedbackSection.style.display = 'none';
    commentBox.value = "";

    const available = await isBackendAvailable();
    if (!available) {
        alert("Backend server is not available. Please make sure it's running.");
        progressContainer.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        progressContainer.style.display = 'none';

        if (!response.ok) throw new Error("Prediction request failed");
        const data = await response.json();

        currentPrediction = data.prediction;
        predictionText.textContent = "Prediction: " + currentPrediction.toUpperCase();
        uploadedImage.src = "data:image/jpeg;base64," + data.image;
        predictionResult.style.display = 'block';
        feedbackSection.style.display = 'block';

    } catch (err) {
        alert("Error during prediction: " + err.message);
        progressContainer.style.display = 'none';
    }
}

async function sendFeedback(decision) {
    if (!currentFile) {
        alert("No image to send feedback for.");
        return;
    }

    const comment = commentBox.value.trim();

    if (comment.length === 0) {
        if (!confirm("You have not entered any comment. Proceed without comment?")) {
            return;
        }
    }

    const formData = new FormData();
    formData.append('prediction', currentPrediction);
    formData.append('decision', decision);
    formData.append('comment', comment);
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${BACKEND_URL}/feedback`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            alert(data.message);
            feedbackSection.style.display = 'none';
            predictionResult.style.display = 'none';
            fileInput.value = "";
            currentFile = null;
            currentPrediction = "";
            commentBox.value = "";
        } else {
            alert("Feedback submission failed: " + (data.message || "Unknown error"));
        }
    } catch (err) {
        alert("Error submitting feedback: " + err.message);
    }
}

acceptBtn.addEventListener('click', () => sendFeedback('accept'));
rejectBtn.addEventListener('click', () => sendFeedback('reject'));
