/* ─── DOM References ───────────────────────────────────── */
const fileInput   = document.getElementById('file-input');
const uploadZone  = document.getElementById('upload-zone');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const btnAnalyse  = document.getElementById('btn-analyse');
const btnText     = document.getElementById('btn-text');
const resultBox   = document.getElementById('result-box');
const resultError = document.getElementById('result-error');

const resGesture  = document.getElementById('res-gesture');
const resEmoji    = document.getElementById('res-emoji');
const resConf     = document.getElementById('res-conf');
const resAcc      = document.getElementById('res-acc');
const resRaw      = document.getElementById('res-raw');
const barConf     = document.getElementById('bar-conf');
const barAcc      = document.getElementById('bar-acc');

/* ─── Emoji map ────────────────────────────────────────── */
const emojiMap = {
    'A':'🤜','B':'✋','C':'🤏','D':'☝️','E':'✊',
    'F':'🤙','G':'👉','H':'🖖','I':'🤞','J':'🤟',
    'K':'✌️','L':'👍','M':'🖐️','N':'✌️','O':'👌',
    'P':'🤙','Q':'👇','R':'🤞','S':'✊','T':'👊',
    'U':'✌️','V':'✌️','W':'🖖','X':'☝️','Y':'🤙','Z':'✍️',
};

/* ─── Drag-and-drop ────────────────────────────────────── */
['dragenter', 'dragover'].forEach(evt => {
    uploadZone.addEventListener(evt, ev => {
        ev.preventDefault();
        uploadZone.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach(evt => {
    uploadZone.addEventListener(evt, ev => {
        ev.preventDefault();
        uploadZone.classList.remove('drag-over');
    });
});

uploadZone.addEventListener('drop', ev => {
    const file = ev.dataTransfer.files[0];
    if (file) loadFile(file);
});

/* ─── File input change ────────────────────────────────── */
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

/* ─── Load & preview file ──────────────────────────────── */
function loadFile(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewWrap.style.display = 'block';
    btnAnalyse.style.display  = 'flex';
    resultBox.style.display   = 'none';
    resultError.style.display = 'none';
    btnAnalyse._file = file;   // store reference for analysis
}

/* ─── Analyse button ───────────────────────────────────── */
btnAnalyse.addEventListener('click', async () => {
    const file = btnAnalyse._file;
    if (!file) return;

    // Set loading state
    btnAnalyse.classList.add('loading');
    btnText.innerHTML = '<span class="spinner"></span> Analysing…';
    resultBox.style.display   = 'none';
    resultError.style.display = 'none';

    try {
        const fd = new FormData();
        fd.append('image', file);

        const resp = await fetch('/predict_image', { method: 'POST', body: fd });
        const data = await resp.json();

        if (!resp.ok || data.error) {
            throw new Error(data.error || 'Server error');
        }

        // Populate predicted gesture
        const gesture = data.gesture.toUpperCase();
        resGesture.textContent = gesture;
        resEmoji.textContent   = emojiMap[gesture] || '🤚';

        // Confidence (sigmoid of SVM score → 0–100 %)
        const conf = Math.min(100, Math.max(0, parseFloat(data.confidence_pct)));
        resConf.textContent = conf.toFixed(1) + '%';
        barConf.style.width = conf + '%';

        // Accuracy score (normalised absolute margin → 0–100 %)
        const acc = Math.min(100, Math.max(0, parseFloat(data.accuracy_pct)));
        resAcc.textContent = acc.toFixed(1) + '%';
        barAcc.style.width = acc + '%';

        // Raw SVM decision value
        resRaw.textContent = parseFloat(data.raw_score).toFixed(6);

        resultBox.style.display = 'flex';

    } catch (err) {
        resultError.textContent   = '⚠️ ' + err.message;
        resultError.style.display = 'block';
    } finally {
        btnAnalyse.classList.remove('loading');
        btnText.innerHTML = '🔍 Analyse Gesture';
    }
});
