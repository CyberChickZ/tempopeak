// ============================================================================
// SAM3 ANNOTATOR
// ============================================================================

window.onerror = function (msg, url, line, col, error) {
    console.error("JS Error: " + msg + "\nLine: " + line);
    return false;
};

let videoEl = document.getElementById('video-player');
let maskCanvas = document.getElementById('mask-canvas');
let uiCanvas = document.getElementById('ui-canvas');
let maskCtx = maskCanvas.getContext('2d');
let uiCtx = uiCanvas.getContext('2d');

let totalFrames = 0;
let fps = 30;
let currentFrame = 0;
let isPlaying = false;
let hasUnsavedChanges = false;

let parsedTracks = {};
let trackBounds = {};

let maskImageCache = new Map();
let frameInstancesCache = new Map();

let idColors = {};

const PALETTE = [
    "#ef4444", "#f59e0b", "#10b981", "#3b82f6",
    "#8b5cf6", "#ec4899", "#f97316", "#14b8a6",
    "#6366f1", "#d946ef", "#06b6d4", "#eab308"
];

function hexToRgba(hex) {
    let r = parseInt(hex.slice(1, 3), 16);
    let g = parseInt(hex.slice(3, 5), 16);
    let b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b];
}

function getColorForId(id) {
    if (!idColors[id]) {
        idColors[id] = PALETTE[Object.keys(idColors).length % PALETTE.length];
    }
    return idColors[id];
}

let hoveredObjectId = null;

// ================= Tab Switching =================
let activeTab = 'annotator';
const tabBtns = document.querySelectorAll('.tab-btn');
const screens = document.querySelectorAll('.screen');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        screens.forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        activeTab = tabId;
        document.getElementById(`${tabId}-screen`).classList.add('active');

        if (tabId === 'clipreview') {
            if (videoEl && !videoEl.paused) videoEl.pause();
            if (haVideo && !haVideo.paused) haVideo.pause();
            crInit();
        } else if (tabId === 'videoclipper') {
            if (videoEl && !videoEl.paused) videoEl.pause();
            if (crPollingTimer) crStopPolling();
        } else if (tabId === 'hitlabeler') {
            if (videoEl && !videoEl.paused) videoEl.pause();
            if (haVideo && !haVideo.paused) { haVideo.pause(); haIsPlaying = false; }
            if (crPollingTimer) crStopPolling();
            if (!hlCanvas) hlInit();
            setTimeout(() => hlDrawCanvas(), 50);
        } else if (tabId === 'personlabel') {
            if (videoEl && !videoEl.paused) videoEl.pause();
            if (haVideo && !haVideo.paused) { haVideo.pause(); haIsPlaying = false; }
            if (crPollingTimer) crStopPolling();
            // Delay to allow layout to settle before drawing canvas
            setTimeout(() => plDrawCanvas(), 50);
        } else {
            if (haVideo && !haVideo.paused) { haVideo.pause(); haIsPlaying = false; }
            if (crPollingTimer) crStopPolling();
        }
    });
});

// ================= Scan and Load Setup =================
document.getElementById('browse-btn').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/browse_dir');
        if (res.ok) {
            const data = await res.json();
            if (data.ok && data.path) {
                document.getElementById('workdir-input').value = data.path;
                scanDirectory();
            }
        }
    } catch (e) { console.error("Browse Error:", e); }
});

window.addEventListener('DOMContentLoaded', () => {
    scanDirectory();
    crInit();
});
document.getElementById('workdir-input').addEventListener('change', scanDirectory);

async function scanDirectory() {
    const workdir = document.getElementById('workdir-input').value.trim();
    if (!workdir) return;

    const listContainer = document.getElementById('video-list');
    listContainer.innerHTML = '<div style="color: #64748b; font-size: 0.85rem; padding: 10px;">Scanning...</div>';

    try {
        const res = await fetch('/api/scan_dir', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workdir })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Backend scan failed");
        }

        const data = await res.json();
        listContainer.innerHTML = '';

        if (data.videos.length === 0) {
            listContainer.innerHTML = '<div style="color: #64748b; font-size: 0.85rem;">No matches found.</div>';
        } else {
            data.videos.forEach(v => {
                const item = document.createElement('div');
                item.className = 'video-list-item';
                item.innerText = v;
                item.onclick = () => selectVideoItem(v, item);
                listContainer.appendChild(item);
            });
            selectVideoItem(data.videos[0], listContainer.children[0]);
        }
    } catch (err) {
        alert(err.message);
        listContainer.innerHTML = '<div style="color: #64748b; font-size: 0.85rem; padding: 10px;">Scan failed.</div>';
    }
}

let selectedVideoName = null;
let activeVideoListItem = null;

async function selectVideoItem(videoName, element) {
    if (selectedVideoName === videoName) return;

    if (hasUnsavedChanges) {
        try {
            await fetch('/api/save_overwrite', { method: 'POST' });
        } catch (e) { console.error("Auto save failed", e); }
    }

    if (activeVideoListItem) activeVideoListItem.classList.remove('active');
    element.classList.add('active');
    activeVideoListItem = element;
    selectedVideoName = videoName;
    hasUnsavedChanges = false;

    loadVideoData(videoName);
}

async function loadVideoData(videoName) {
    if (isPlaying) {
        videoEl.pause();
        isPlaying = false;
        document.getElementById('play-pause-btn').innerText = "▶";
    }

    fps = parseInt(document.getElementById('fps-input').value) || 30;
    document.getElementById('loading-msg').innerText = "Loading data from backend...";

    try {
        const res = await fetch('/api/load_by_name', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_name: videoName })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Backend load failed");
        }

        videoEl.src = `/video?vid=${videoName}&t=${Date.now()}`;

        const dlRes = await fetch('/api/download_json');
        const jsonText = await dlRes.text();
        parsedTracks = JSON.parse(jsonText);
        delete parsedTracks["_meta"];
        buildTimelineData();

        videoEl.onloadedmetadata = () => {
            totalFrames = Math.max(...Object.keys(parsedTracks).map(Number), 0);

            if (videoEl.duration > 0 && totalFrames > 0) {
                const calculatedFps = Math.round(totalFrames / videoEl.duration);
                if (calculatedFps > 0) {
                    fps = calculatedFps;
                    document.getElementById('fps-input').value = fps;
                }
            }

            maskCanvas.width = videoEl.videoWidth;
            maskCanvas.height = videoEl.videoHeight;
            uiCanvas.width = videoEl.videoWidth;
            uiCanvas.height = videoEl.videoHeight;

            const wrapper = document.querySelector('.video-wrapper');
            wrapper.style.aspectRatio = `${videoEl.videoWidth} / ${videoEl.videoHeight}`;

            renderTimelineTracks();

            let targetFrame = 0;
            let maxScore = -1;
            for (let f = 0; f <= totalFrames; f++) {
                if (parsedTracks[f]) {
                    Object.values(parsedTracks[f]).forEach(info => {
                        if (info.hit_score !== undefined && info.hit_score > maxScore) {
                            maxScore = info.hit_score;
                            targetFrame = f;
                        }
                    });
                }
            }
            setFrame(targetFrame);

            document.getElementById('loading-msg').innerText = "Loaded successfully!";
            setTimeout(() => { document.getElementById('loading-msg').innerText = ""; }, 3000);
        };

        if (videoEl.readyState >= 1) videoEl.onloadedmetadata(null);

    } catch (err) {
        alert(err.message);
        document.getElementById('loading-msg').innerText = "Load failed.";
    }
}

// ================= Timeline & Data Processing =================
function buildTimelineData() {
    trackBounds = {};
    const frames = Object.keys(parsedTracks).map(Number).sort((a, b) => a - b);

    const objIds = new Set();
    frames.forEach(f => Object.keys(parsedTracks[f]).forEach(oid => objIds.add(oid)));

    objIds.forEach(oid => {
        let presenceObjFrames = [];
        frames.forEach(f => { if (parsedTracks[f][oid]) presenceObjFrames.push(f); });

        let intervals = [];
        if (presenceObjFrames.length > 0) {
            let start = presenceObjFrames[0], end = start;
            for (let i = 1; i < presenceObjFrames.length; i++) {
                if (presenceObjFrames[i] === end + 1) {
                    end = presenceObjFrames[i];
                } else {
                    intervals.push([start, end]);
                    start = presenceObjFrames[i];
                    end = start;
                }
            }
            intervals.push([start, end]);
        }

        trackBounds[oid] = { intervals, defaultLabel: "unknown" };

        for (let f of frames) {
            if (parsedTracks[f][oid]) {
                trackBounds[oid].defaultLabel = parsedTracks[f][oid].label || parsedTracks[f][oid].prompt;
                break;
            }
        }
    });
}

function renderTimelineTracks() {
    const container = document.getElementById('tracks-container');
    container.innerHTML = '';

    Object.keys(trackBounds).forEach(oid => {
        const row = document.createElement('div');
        row.className = 'track-row';

        const label = document.createElement('div');
        label.className = 'track-label';
        label.innerText = `Obj ${oid} (${trackBounds[oid].defaultLabel})`;

        const bg = document.createElement('div');
        bg.className = 'track-bg';

        row.appendChild(bg);
        row.appendChild(label);

        const colorHex = getColorForId(oid);

        trackBounds[oid].intervals.forEach(interval => {
            const leftPct = (interval[0] / totalFrames) * 100;
            const widthPct = ((interval[1] - interval[0] + 1) / totalFrames) * 100;
            const bar = document.createElement('div');
            bar.className = 'track-presence';
            bar.style.left = `${leftPct}%`;
            bar.style.width = `${widthPct}%`;
            bar.style.backgroundColor = colorHex;
            row.appendChild(bar);
        });

        let hasHitScore = false;
        let points = [];
        for (let f = 0; f <= totalFrames; f++) {
            if (parsedTracks[f] && parsedTracks[f][oid]) {
                const hs = parsedTracks[f][oid].hit_score;
                if (hs !== undefined) { hasHitScore = true; points.push({ f, s: hs }); }
            }
        }

        if (hasHitScore && points.length > 0) {
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;";
            svg.setAttribute("preserveAspectRatio", "none");
            svg.setAttribute("viewBox", "0 0 1000 100");
            let d = "";
            let started = false;
            for (let p of points) {
                const x = (p.f / totalFrames) * 1000;
                const y = 100 - (p.s * 100);
                d += started ? `L ${x} ${y} ` : `M ${x} ${y} `;
                started = true;
            }
            if (d) {
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("d", d);
                path.setAttribute("fill", "none");
                path.setAttribute("stroke", "#3b82f6");
                path.setAttribute("stroke-width", "3");
                path.setAttribute("vector-effect", "non-scaling-stroke");
                svg.appendChild(path);
                row.appendChild(svg);
            }
        }

        container.appendChild(row);
    });

    container.addEventListener('click', (e) => {
        const rect = container.getBoundingClientRect();
        const clickedFrame = Math.round(((e.clientX - rect.left) / rect.width) * totalFrames);
        setFrame(Math.min(Math.max(clickedFrame, 0), totalFrames));
    });
}

// ================= Playback & Navigation =================
function setFrame(frame) {
    if (frame < 0) frame = 0;
    if (frame > totalFrames) frame = totalFrames;
    videoEl.currentTime = (frame / fps) + 0.005;
    onFrameChange(frame);
}

function playbackLoop() {
    if (!isPlaying) return;
    const frame = Math.round(videoEl.currentTime * fps);
    if (frame !== currentFrame && frame <= totalFrames) onFrameChange(frame);
    if (!videoEl.paused && !videoEl.ended) requestAnimationFrame(playbackLoop);
}

videoEl.addEventListener('ended', () => {
    isPlaying = false;
    document.getElementById('play-pause-btn').innerText = "▶";
});

document.getElementById('play-pause-btn').addEventListener('click', () => {
    if (videoEl.paused) {
        videoEl.play().catch(e => console.warn("Play interrupted", e));
        isPlaying = true;
        document.getElementById('play-pause-btn').innerText = "⏸";
        requestAnimationFrame(playbackLoop);
    } else {
        videoEl.pause();
        isPlaying = false;
        document.getElementById('play-pause-btn').innerText = "▶";
        setFrame(currentFrame);
    }
});

document.getElementById('prev-btn').addEventListener('click', () => setFrame(currentFrame - 1));
document.getElementById('next-btn').addEventListener('click', () => setFrame(currentFrame + 1));
document.getElementById('frame-input').addEventListener('change', (e) => {
    const val = parseInt(e.target.value);
    if (!isNaN(val)) setFrame(val);
});

document.getElementById('fps-input').addEventListener('change', (e) => {
    const val = parseInt(e.target.value);
    if (!isNaN(val) && val > 0) {
        const currentTimeSec = videoEl.currentTime || 0;
        fps = val;
        onFrameChange(Math.round(currentTimeSec * fps));
    }
});

async function applyHitScoreCalibration() {
    try {
        const res = await fetch('/api/edit_hit_score_gaussian', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: currentFrame, sigma: 2.5 })
        });
        if (res.ok) {
            const dlRes = await fetch('/api/download_json');
            const jsonText = await dlRes.text();
            parsedTracks = JSON.parse(jsonText);
            delete parsedTracks["_meta"];
            buildTimelineData();
            renderTimelineTracks();
            onFrameChange(currentFrame);
            hasUnsavedChanges = true;
        }
    } catch (err) { console.error("Error applying hit score:", err); }
}

document.getElementById('calibrate-btn').addEventListener('click', applyHitScoreCalibration);

// SAM3 Keyboard (annotator tab only)
document.addEventListener('keydown', async e => {
    if (activeTab !== 'annotator') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'a' || e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    if (e.key === 'd' || e.key === 'ArrowRight') setFrame(currentFrame + 1);
    if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (activeVideoListItem && activeVideoListItem.previousElementSibling)
            activeVideoListItem.previousElementSibling.click();
    }
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (activeVideoListItem && activeVideoListItem.nextElementSibling)
            activeVideoListItem.nextElementSibling.click();
    }
    if (e.key === ' ') { e.preventDefault(); await applyHitScoreCalibration(); }
});

function updatePlayheadUI() {
    const pct = (currentFrame / totalFrames) * 100;
    document.getElementById('playhead').style.left = `${pct}%`;
    document.getElementById('frame-input').value = currentFrame;
    document.getElementById('current-frame-lbl').innerText = currentFrame;
}

// ================= Core Rendering & Fetching =================
async function onFrameChange(frame) {
    currentFrame = frame;
    updatePlayheadUI();
    let instances = {};
    if (parsedTracks && parsedTracks[frame]) instances = parsedTracks[frame];
    renderSidebar(instances);
    await drawMasks(instances);
}

async function fetchMaskImage(mask_idx) {
    if (maskImageCache.has(mask_idx)) return maskImageCache.get(mask_idx);
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => { maskImageCache.set(mask_idx, img); resolve(img); };
        img.onerror = reject;
        img.src = `/api/mask/${mask_idx}.png`;
    });
}

function tintCanvas(img, rgbArray, outCanvas, highlight = false) {
    outCanvas.width = img.width;
    outCanvas.height = img.height;
    const ctx = outCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    ctx.globalCompositeOperation = 'source-in';
    const [r, g, b] = rgbArray;
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${highlight ? 0.7 : 0.4})`;
    ctx.fillRect(0, 0, img.width, img.height);
    ctx.globalCompositeOperation = 'source-over';
    return outCanvas;
}

async function drawMasks(instances) {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);

    const tempCanvas = document.createElement('canvas');
    const promises = Object.entries(instances).map(async ([objId, info]) => {
        try {
            const img = await fetchMaskImage(info.mask_idx);
            return { objId, info, img };
        } catch (e) { return null; }
    });

    const results = (await Promise.all(promises)).filter(r => r !== null);

    for (const { objId, info, img } of results) {
        const colorHex = getColorForId(objId);
        const color = hexToRgba(colorHex);
        const isHovered = hoveredObjectId === objId;
        maskCtx.drawImage(tintCanvas(img, color, tempCanvas, isHovered), 0, 0);

        if (info.box) {
            uiCtx.strokeStyle = `rgb(${color.join(',')})`;
            uiCtx.globalAlpha = isHovered ? 1.0 : 0.4;
            uiCtx.lineWidth = isHovered ? 2 : 1;
            uiCtx.strokeRect(info.box[0], info.box[1], info.box[2] - info.box[0], info.box[3] - info.box[1]);
            uiCtx.globalAlpha = 1.0;
        }
    }
}

// ================= Sidebar & Editing =================
function renderSidebar(instances) {
    const list = document.getElementById('objects-list');
    list.innerHTML = '';

    const entries = Object.entries(instances);
    if (entries.length === 0) {
        list.innerHTML = '<div class="empty-state">No objects in this frame.</div>';
        return;
    }

    for (const [objId, info] of entries) {
        const card = document.createElement('div');
        card.className = 'obj-card';
        if (hoveredObjectId === objId) card.classList.add('highlight');

        const colorHex = getColorForId(objId);

        card.innerHTML = `
            <div class="obj-header">
                <div class="obj-id">
                    <span class="color-dot" style="background: ${colorHex}"></span>
                    Object ID: ${objId}
                </div>
                <button class="obj-delete-track" title="Delete object from all frames">Del Track</button>
                <button class="obj-delete" title="Delete Instance">✕</button>
            </div>
            <div class="obj-body">
                <select class="label-select">
                    ${(() => {
                const baseKeys = ['ball', 'racket'];
                const discoveredKeys = [...new Set(Object.values(trackBounds).map(b => b.defaultLabel))].filter(l => l && l !== 'unknown');
                const allKeys = [...new Set([...baseKeys, ...discoveredKeys])];
                let html = '';
                let hasMatched = false;
                allKeys.forEach(k => {
                    const isSelected = (info.label || info.prompt) === k;
                    if (isSelected) hasMatched = true;
                    html += `<option value="${k}" ${isSelected ? 'selected' : ''}>${k}</option>`;
                });
                html += `<option value="unknown" ${!hasMatched || (info.label || info.prompt) === 'unknown' ? 'selected' : ''}>Unknown</option>`;
                return html;
            })()}
                </select>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Score: ${(info.tracker_score ?? info.score ?? 0).toFixed(3)}</div>
                ${info.hit_score !== undefined ? `<div style="font-size: 0.8rem; color: #ef4444; font-weight: bold; margin-top: 2px;">Hit Score: ${(info.hit_score).toFixed(3)}</div>` : ''}
            </div>
        `;

        card.addEventListener('mouseenter', () => {
            hoveredObjectId = objId;
            card.classList.add('highlight');
            drawMasks(instances);
        });
        card.addEventListener('mouseleave', () => {
            hoveredObjectId = null;
            card.classList.remove('highlight');
            drawMasks(instances);
        });

        const select = card.querySelector('.label-select');
        select.addEventListener('change', async (e) => {
            const newLabel = e.target.value;
            if ('label' in info) info.label = newLabel; else info.prompt = newLabel;
            drawMasks(instances);

            await fetch('/api/edit_track', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ object_id: objId, prompt: newLabel })
            });

            for (const f in parsedTracks) {
                if (parsedTracks[f][objId]) {
                    if ('label' in parsedTracks[f][objId]) parsedTracks[f][objId].label = newLabel;
                    else parsedTracks[f][objId].prompt = newLabel;
                }
            }
            if (trackBounds[objId]) trackBounds[objId].defaultLabel = newLabel;
            hasUnsavedChanges = true;
        });

        const delBtn = card.querySelector('.obj-delete');
        delBtn.addEventListener('click', async () => {
            await fetch('/api/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_idx: currentFrame, object_id: objId })
            });
            delete parsedTracks[currentFrame][objId];
            hasUnsavedChanges = true;
            buildTimelineData();
            renderTimelineTracks();
            onFrameChange(currentFrame);
        });

        const delTrackBtn = card.querySelector('.obj-delete-track');
        let deleteConfirmTimeout;
        delTrackBtn.addEventListener('click', async () => {
            if (delTrackBtn.innerText === 'Del Track') {
                delTrackBtn.innerText = 'Sure?';
                delTrackBtn.style.backgroundColor = 'var(--danger)';
                delTrackBtn.style.color = '#fff';
                deleteConfirmTimeout = setTimeout(() => {
                    delTrackBtn.innerText = 'Del Track';
                    delTrackBtn.style.backgroundColor = '';
                    delTrackBtn.style.color = '';
                }, 3000);
                return;
            }
            clearTimeout(deleteConfirmTimeout);
            await fetch('/api/delete_track', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ object_id: objId })
            });
            for (const frame in parsedTracks) {
                if (parsedTracks[frame][objId]) delete parsedTracks[frame][objId];
            }
            hasUnsavedChanges = true;
            buildTimelineData();
            renderTimelineTracks();
            onFrameChange(currentFrame);
        });

        list.appendChild(card);
    }
}

// ================= Save =================
document.getElementById('save-overwrite-btn').addEventListener('click', async () => {
    const btn = document.getElementById('save-overwrite-btn');
    btn.innerText = "Saving...";
    try {
        const res = await fetch('/api/save_overwrite', { method: 'POST' });
        if (res.ok) {
            hasUnsavedChanges = false;
            btn.innerText = "Saved!";
            setTimeout(() => { btn.innerText = "Save (Overwrite)"; }, 2000);
        } else {
            alert("Failed to save. Check server logs.");
            btn.innerText = "Save (Overwrite)";
        }
    } catch (e) {
        console.error(e);
        btn.innerText = "Save (Overwrite)";
    }
});

document.getElementById('save-as-btn').addEventListener('click', () => {
    window.open('/api/download_json', '_blank');
    setTimeout(() => {
        window.open('/api/download_npz', '_blank');
        hasUnsavedChanges = false;
    }, 500);
});


// ============================================================================
// CLIP REVIEW MODULE
// ============================================================================

let crClips = [];
let crCurrentIdx = -1;
let crCurrentFrame = 0;
let crPollingTimer = null;
let crProcessing = false;

async function crInit() {
    try {
        const lf = await fetch('/api/clips/last_folder').then(r => r.json());
        if (lf.folder) document.getElementById('clip-folder-input').value = lf.folder;
    } catch (e) { console.error('last_folder:', e); }

    try {
        const { clips, processing } = await fetch('/api/clips/list').then(r => r.json());
        crProcessing = processing;
        crSyncList(clips);
        if (processing && !crPollingTimer) crStartPolling();
        if (crCurrentIdx === -1 && crClips.length > 0) {
            const first = crClips.findIndex(c => !c.annotated);
            crSelectClip(first !== -1 ? first : 0);
        }
    } catch (e) { console.error('crInit:', e); }
}

function crStartPolling() {
    crPollingTimer = setInterval(async () => {
        try {
            const { clips, processing } = await fetch('/api/clips/list').then(r => r.json());
            crProcessing = processing;
            crSyncList(clips);
            document.getElementById('clip-progress').textContent = processing
                ? `处理中 ${clips.length} clips…`
                : `处理完成 共 ${clips.length} clips`;
            if (!processing) crStopPolling();
        } catch (e) { console.error('polling:', e); }
    }, 2000);
}

function crStopPolling() {
    clearInterval(crPollingTimer);
    crPollingTimer = null;
}

function crSyncList(newClips) {
    newClips.forEach(c => {
        // Use DOM as source of truth to avoid duplicate appends
        if (!document.getElementById(`clip-item-${c.clip_id}`)) {
            crClips.push(c);
            crAppendListItem(c, crClips.length - 1);
            if (crCurrentIdx === -1) crSelectClip(0);
        } else {
            const local = crClips.find(x => x.clip_id === c.clip_id);
            if (local) {
                Object.assign(local, c);
                crUpdateListItem(crClips.indexOf(local));
            }
        }
    });
}

function crAppendListItem(clip, idx) {
    const list = document.getElementById('clip-list');
    const div = document.createElement('div');
    div.id = `clip-item-${clip.clip_id}`;
    div.className = 'clip-list-item';
    div.dataset.clipIdx = idx;
    const shortName = (clip.clip_id.match(/clip_\d+(?!.*clip_\d)/) || [clip.clip_id])[0];
    div.textContent = clip.annotated ? `✓ ${shortName}` : shortName;
    if (clip.annotated) { div.classList.add('annotated'); div.title = `Hit: f${clip.hit_frame}`; }
    div.onclick = () => crSelectClip(idx);
    list.appendChild(div);
}

function crUpdateListItem(idx) {
    const items = document.getElementById('clip-list').querySelectorAll('.clip-list-item');
    if (!items[idx]) return;
    const clip = crClips[idx];
    const shortName = (clip.clip_id.match(/clip_\d+(?!.*clip_\d)/) || [clip.clip_id])[0];
    items[idx].textContent = clip.annotated ? `✓ ${shortName}` : shortName;
    items[idx].classList.toggle('annotated', clip.annotated);
    items[idx].classList.toggle('active', idx === crCurrentIdx);
    items[idx].title = clip.annotated ? `Hit: f${clip.hit_frame}` : '';
}

function crHighlightListItem(idx) {
    const items = document.getElementById('clip-list').querySelectorAll('.clip-list-item');
    items.forEach((item, i) => item.classList.toggle('active', i === idx));
    if (items[idx]) items[idx].scrollIntoView({ block: 'nearest' });
}

function crSelectClip(idx) {
    if (idx < 0 || idx >= crClips.length) return;
    crCurrentIdx = idx;
    crCurrentFrame = 0;
    const clip = crClips[idx];
    crHighlightListItem(idx);
    crRenderFilmstrip(clip);
    document.getElementById('clip-info').textContent =
        `${clip.clip_id} | frame 1 / ${clip.num_frames} | 按空格标记击球帧`;
    const startFrame = clip.annotated ? clip.hit_frame : (clip.peak_frame ?? 0);
    crRenderFrame(clip.clip_id, startFrame);
}

function crRenderFrame(clip_id, frame_idx) {
    const clip = crClips[crCurrentIdx];
    if (!clip) return;
    if (frame_idx < 0) frame_idx = 0;
    if (frame_idx >= clip.num_frames) frame_idx = clip.num_frames - 1;
    document.getElementById('frame-display').src =
        `/api/clips/frame/${clip_id}/${frame_idx}?t=${Date.now()}`;
    crCurrentFrame = frame_idx;
    document.getElementById('clip-info').textContent =
        `${clip.clip_id} | frame ${frame_idx + 1} / ${clip.num_frames}`;
    crHighlightFilmstrip(frame_idx);
}

function crRenderFilmstrip(clip) {
    const strip = document.getElementById('clip-filmstrip');
    strip.innerHTML = '';
    for (let i = 0; i < clip.num_frames; i++) {
        const img = document.createElement('img');
        img.src = `/api/clips/frame/${clip.clip_id}/${i}`;
        img.dataset.frame = i;
        img.onclick = () => crRenderFrame(clip.clip_id, i);
        if (clip.annotated && clip.hit_frame === i) img.classList.add('hit-frame');
        strip.appendChild(img);
    }
}

function crHighlightFilmstrip(frame_idx) {
    const imgs = document.getElementById('clip-filmstrip').querySelectorAll('img');
    imgs.forEach(img => img.classList.toggle('active', img.dataset.frame === String(frame_idx)));
    const active = document.getElementById('clip-filmstrip').querySelector('img.active');
    if (active) active.scrollIntoView({ inline: 'nearest', block: 'nearest' });
}

function crNextUnannotated() {
    for (let i = crCurrentIdx + 1; i < crClips.length; i++) {
        if (!crClips[i].annotated) return i;
    }
    return crCurrentIdx;
}

async function crMarkReject() {
    if (crCurrentIdx < 0) return;
    const clip = crClips[crCurrentIdx];
    try {
        await fetch('/api/clips/reject', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clip_id: clip.clip_id })
        });
        const list = document.getElementById('clip-list');
        if (list.children[crCurrentIdx]) list.removeChild(list.children[crCurrentIdx]);
        crClips.splice(crCurrentIdx, 1);
        Array.from(list.children).forEach((item, i) => {
            item.dataset.clipIdx = i;
            item.onclick = () => crSelectClip(i);
        });
        if (crClips.length === 0) {
            document.getElementById('frame-display').src = '';
            document.getElementById('clip-filmstrip').innerHTML = '';
            document.getElementById('clip-info').textContent = 'No clip loaded';
        } else {
            crSelectClip(Math.min(crCurrentIdx, crClips.length - 1));
        }
    } catch (e) { console.error('reject:', e); }
}

async function crMarkUndo() {
    if (crCurrentIdx < 0) return;
    const clip = crClips[crCurrentIdx];
    try {
        await fetch('/api/clips/undo', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clip_id: clip.clip_id })
        });
        clip.annotated = false;
        clip.hit_frame = null;
        crUpdateListItem(crCurrentIdx);
        document.getElementById('clip-filmstrip').querySelectorAll('img').forEach(img => {
            img.classList.remove('hit-frame');
        });
    } catch (e) { console.error('undo:', e); }
}

// ── Browse ──────────────────────────────────────────────────────────
document.getElementById('clip-browse-btn').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/browse_dir');
        const data = await res.json();
        if (data.ok && data.path) {
            document.getElementById('clip-folder-input').value = data.path;
        }
    } catch (e) { console.error('browse:', e); }
});

// ── Start ───────────────────────────────────────────────────────────
document.getElementById('clip-start-btn').addEventListener('click', async () => {
    const folder = document.getElementById('clip-folder-input').value.trim();
    if (!folder) return;
    const params = {
        folder,
        hit_thresh: parseFloat(document.getElementById('p-hit-thresh').value) || 150,
        pad_before: parseInt(document.getElementById('p-pad-before').value) || 15,
        pad_after: parseInt(document.getElementById('p-pad-after').value) || 15,
        export_pad: parseInt(document.getElementById('p-export-pad').value) || 32,
        conf: parseFloat(document.getElementById('p-conf').value) || 0.25,
        iou: parseFloat(document.getElementById('p-iou').value) || 0.7,
        imgsz: parseInt(document.getElementById('p-imgsz').value) || 640,
        show_boxes: document.getElementById('p-show-boxes').checked,
    };
    try {
        const res = await fetch('/api/clips/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        if (!res.ok) { console.error('start failed:', (await res.json()).detail); return; }
        crClips = []; crCurrentIdx = -1; crCurrentFrame = 0;
        document.getElementById('clip-list').innerHTML = '';
        document.getElementById('clip-progress').textContent = 'Starting...';
        document.getElementById('frame-display').src = '';
        document.getElementById('clip-filmstrip').innerHTML = '';
        document.getElementById('clip-info').textContent = '检测中，请等待...';
        if (crPollingTimer) crStopPolling();
        crStartPolling();
    } catch (e) { console.error('clip-start:', e); }
});

// ── Clear ────────────────────────────────────────────────────────────
document.getElementById('clip-clear-btn').addEventListener('click', async () => {
    if (!confirm('清除所有 clip 结果？')) return;
    try {
        const res = await fetch('/api/clips/clear', { method: 'POST' });
        if (!res.ok) { console.error('clear failed:', (await res.json()).detail); return; }
        crClips = []; crCurrentIdx = -1; crCurrentFrame = 0;
        document.getElementById('clip-list').innerHTML = '';
        document.getElementById('frame-display').src = '';
        document.getElementById('clip-filmstrip').innerHTML = '';
        document.getElementById('clip-info').textContent = 'No clip loaded';
        document.getElementById('clip-progress').textContent = 'Ready';
        if (crPollingTimer) crStopPolling();
    } catch (e) { console.error('clear:', e); }
});

// ── Export ──────────────────────────────────────────────────────────
document.getElementById('clip-export-btn').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/clips/export', { method: 'POST' });
        const data = await res.json();
        if (data.ok) alert(`导出完成：${data.exported} clips → ${data.out_dir}`);
        else console.error('export failed:', data.detail);
    } catch (e) { console.error('export:', e); }
});

// ── Keyboard (clip review tab only) ─────────────────────────────────
document.addEventListener('keydown', async e => {
    if (activeTab !== 'clipreview') return;
    if (e.target.tagName === 'INPUT') return;

    const clip = crClips[crCurrentIdx];

    if (e.key === 'ArrowRight') {
        if (clip) crRenderFrame(clip.clip_id, crCurrentFrame + 1);
        e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
        if (clip) crRenderFrame(clip.clip_id, crCurrentFrame - 1);
        e.preventDefault();
    } else if (e.key === 'ArrowDown') {
        crSelectClip(crCurrentIdx + 1); e.preventDefault();
    } else if (e.key === 'ArrowUp') {
        crSelectClip(crCurrentIdx - 1); e.preventDefault();
    } else if (e.key === ' ') {
        e.preventDefault();
        if (crCurrentIdx < 0 || !clip) return;
        try {
            await fetch('/api/clips/annotate', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clip_id: clip.clip_id, hit_frame: crCurrentFrame })
            });
            clip.annotated = true; clip.hit_frame = crCurrentFrame;
            crUpdateListItem(crCurrentIdx);
            document.getElementById('clip-filmstrip').querySelectorAll('img').forEach(img => {
                img.classList.toggle('hit-frame', img.dataset.frame === String(crCurrentFrame));
            });
            crSelectClip(crNextUnannotated());
        } catch (err) { console.error('annotate:', err); }
    } else if (e.key === 'Delete') {
        crMarkReject(); e.preventDefault();
    } else if (e.key === 'Backspace') {
        crMarkUndo(); e.preventDefault();
    }
});


// ============================================================================
// HIT ANNOTATOR MODULE
// ============================================================================

let haVideo = null;
let haCanvas = null;
let haCtx = null;
let haFps = 30;
let haTotalFrames = 0;
let haCurrentFrame = 0;
let haIsPlaying = false;
let haHits = [];             // sorted frame indices
let haCrops = [];            // sorted cut-point frame indices (unlimited)
let haIgnoredSegments = new Set(); // set of segment indices to exclude
let haVideoFile = null;  // File object from input
let haVideoFileName = ''; // original file name
let haHistory = [];          // undo stack [{hits, crops, ignoredSegments}]
let haIsSeeking = false;     // true while a seek is in flight
let haPendingFrame = null;   // latest frame requested while a seek was in progress

function haSaveHistory() {
    haHistory.push({
        hits: [...haHits],
        crops: [...haCrops],
        ignoredSegments: new Set(haIgnoredSegments)
    });
    if (haHistory.length > 50) haHistory.shift();
}

function haUndo() {
    if (haHistory.length === 0) return;
    const prev = haHistory.pop();
    haHits = prev.hits;
    haCrops = prev.crops;
    haIgnoredSegments = prev.ignoredSegments;
    haUpdateUI();
}

// Pending-seek: ensures the video frame always catches up to the latest requested frame
function haSeekTo(frame) {
    if (haIsSeeking) {
        haPendingFrame = frame;
    } else {
        haIsSeeking = true;
        haPendingFrame = null;
        haVideo.currentTime = frame / haFps + 0.001;
    }
}

function haInit() {
    haVideo = document.getElementById('ha-video');
    haCanvas = document.getElementById('ha-timeline-canvas');
    haCtx = haCanvas.getContext('2d');

    // File input
    document.getElementById('ha-file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        haVideoFile = file;
        haVideoFileName = file.name;
        haVideo.src = URL.createObjectURL(file);
        haHits = [];
        haCrops = [];
        haIgnoredSegments = new Set();
        haHistory = [];
        haIsSeeking = false;
        haPendingFrame = null;
        haCurrentFrame = 0;
        document.getElementById('ha-video-info').textContent = `Loading ${file.name}...`;
    });

    // Pending-seek: when a seek completes, immediately seek to the latest pending frame if any
    haVideo.addEventListener('seeked', () => {
        if (haPendingFrame !== null) {
            const frame = haPendingFrame;
            haPendingFrame = null;
            haCurrentFrame = frame;
            haVideo.currentTime = frame / haFps + 0.001;
            haUpdateFrameUI();
        } else {
            haIsSeeking = false;
        }
    });

    haVideo.addEventListener('loadedmetadata', () => {
        haFps = parseInt(document.getElementById('ha-fps-input').value) || 30;
        haTotalFrames = Math.round(haVideo.duration * haFps);
        document.getElementById('ha-total-frames').textContent = haTotalFrames;
        document.getElementById('ha-video-info').textContent =
            `${haVideoFileName} | ${haTotalFrames} frames | ${haVideo.videoWidth}×${haVideo.videoHeight}`;
        haSetFrame(0);
    });

    haVideo.addEventListener('ended', () => {
        haIsPlaying = false;
        document.getElementById('ha-play-btn').innerText = "▶";
    });

    // Play/Pause
    document.getElementById('ha-play-btn').addEventListener('click', () => {
        if (haVideo.paused) {
            haVideo.play().catch(() => { });
            haIsPlaying = true;
            document.getElementById('ha-play-btn').innerText = "⏸";
            requestAnimationFrame(haPlaybackLoop);
        } else {
            haVideo.pause();
            haIsPlaying = false;
            document.getElementById('ha-play-btn').innerText = "▶";
            haSetFrame(haCurrentFrame);
        }
    });

    document.getElementById('ha-prev-btn').addEventListener('click', () => haSetFrame(haCurrentFrame - 1));
    document.getElementById('ha-next-btn').addEventListener('click', () => haSetFrame(haCurrentFrame + 1));
    document.getElementById('ha-frame-input').addEventListener('change', (e) => {
        const v = parseInt(e.target.value);
        if (!isNaN(v)) haSetFrame(v);
    });
    document.getElementById('ha-fps-input').addEventListener('change', () => {
        haFps = parseInt(document.getElementById('ha-fps-input').value) || 30;
        haTotalFrames = Math.round(haVideo.duration * haFps);
        document.getElementById('ha-total-frames').textContent = haTotalFrames;
    });

    // Timeline click
    document.getElementById('ha-timeline-wrap').addEventListener('click', (e) => {
        if (haTotalFrames <= 0) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        haSetFrame(Math.round(pct * haTotalFrames));
    });

    // Resize observer for canvas
    const ro = new ResizeObserver(() => haDrawTimeline());
    ro.observe(document.getElementById('ha-timeline-wrap'));

    // Save
    document.getElementById('ha-save-btn').addEventListener('click', haSave);

    // Clear
    document.getElementById('ha-clear-btn').addEventListener('click', () => {
        haSaveHistory();
        haHits = [];
        haCrops = [];
        haIgnoredSegments = new Set();
        haUpdateUI();
    });

    // Browse
    document.getElementById('ha-browse-btn').addEventListener('click', async () => {
        try {
            const res = await fetch('/api/browse_dir');
            const data = await res.json();
            if (data.ok && data.path) {
                document.getElementById('ha-workdir-input').value = data.path;
            }
        } catch (e) { console.error('browse:', e); }
    });
}

function haPlaybackLoop() {
    if (!haIsPlaying) return;
    const frame = Math.round(haVideo.currentTime * haFps);
    if (frame !== haCurrentFrame && frame <= haTotalFrames) {
        haCurrentFrame = frame;
        haUpdateFrameUI();
    }
    if (!haVideo.paused && !haVideo.ended) requestAnimationFrame(haPlaybackLoop);
}

function haSetFrame(frame) {
    if (frame < 0) frame = 0;
    if (frame > haTotalFrames) frame = haTotalFrames;
    haCurrentFrame = frame;
    haSeekTo(frame);
    haUpdateFrameUI();
}

function haUpdateFrameUI() {
    document.getElementById('ha-frame-input').value = haCurrentFrame;
    haDrawTimeline();
}

function haUpdateUI() {
    haUpdateFrameUI();
    haRenderHitList();
    haRenderCropList();
}

function haDrawTimeline() {
    const wrap = document.getElementById('ha-timeline-wrap');
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    haCanvas.width = w * devicePixelRatio;
    haCanvas.height = h * devicePixelRatio;
    haCanvas.style.width = w + 'px';
    haCanvas.style.height = h + 'px';
    haCtx.scale(devicePixelRatio, devicePixelRatio);

    // Background
    haCtx.fillStyle = '#e2e8f0';
    haCtx.fillRect(0, 0, w, h);

    if (haTotalFrames <= 0) return;

    // Segment shading: build boundary array [0, ...haCrops, haTotalFrames]
    const boundaries = [0, ...haCrops, haTotalFrames];
    for (let i = 0; i < boundaries.length - 1; i++) {
        const sx = (boundaries[i] / haTotalFrames) * w;
        const ex = (boundaries[i + 1] / haTotalFrames) * w;
        haCtx.fillStyle = haIgnoredSegments.has(i) ? 'rgba(239,68,68,0.18)' : 'rgba(219,234,254,0.7)';
        haCtx.fillRect(sx, 0, ex - sx, h);
    }

    // Cut-point markers (blue lines)
    haCtx.strokeStyle = '#2563eb';
    haCtx.lineWidth = 2;
    haCrops.forEach(f => {
        const x = (f / haTotalFrames) * w;
        haCtx.beginPath();
        haCtx.moveTo(x, 0);
        haCtx.lineTo(x, h);
        haCtx.stroke();
        // Label "C" at top
        haCtx.fillStyle = '#2563eb';
        haCtx.font = 'bold 10px sans-serif';
        haCtx.fillText('C', x + 3, 11);
    });

    // Ignored segment "X" label in center of each ignored segment
    haCtx.font = 'bold 12px sans-serif';
    haCtx.fillStyle = 'rgba(239,68,68,0.8)';
    haIgnoredSegments.forEach(i => {
        if (i < boundaries.length - 1) {
            const cx = ((boundaries[i] + boundaries[i + 1]) / 2 / haTotalFrames) * w;
            haCtx.fillText('✕', cx - 5, h / 2 + 5);
        }
    });

    // HIT markers (red triangles + lines)
    haCtx.fillStyle = '#ef4444';
    haCtx.strokeStyle = '#ef4444';
    haCtx.lineWidth = 1.5;
    haHits.forEach(f => {
        const x = (f / haTotalFrames) * w;
        // Line
        haCtx.beginPath();
        haCtx.moveTo(x, 0);
        haCtx.lineTo(x, h);
        haCtx.stroke();
        // Triangle at top
        haCtx.beginPath();
        haCtx.moveTo(x, 0);
        haCtx.lineTo(x - 5, 10);
        haCtx.lineTo(x + 5, 10);
        haCtx.closePath();
        haCtx.fill();
    });

    // Playhead (white line)
    const px = (haCurrentFrame / haTotalFrames) * w;
    haCtx.strokeStyle = '#ffffff';
    haCtx.lineWidth = 2;
    haCtx.beginPath();
    haCtx.moveTo(px, 0);
    haCtx.lineTo(px, h);
    haCtx.stroke();

    // Playhead shadow for visibility
    haCtx.strokeStyle = 'rgba(0,0,0,0.3)';
    haCtx.lineWidth = 4;
    haCtx.beginPath();
    haCtx.moveTo(px, 0);
    haCtx.lineTo(px, h);
    haCtx.stroke();
    // White on top
    haCtx.strokeStyle = '#ffffff';
    haCtx.lineWidth = 2;
    haCtx.beginPath();
    haCtx.moveTo(px, 0);
    haCtx.lineTo(px, h);
    haCtx.stroke();
}

function haRenderHitList() {
    const el = document.getElementById('ha-hit-list');
    el.innerHTML = '';
    haHits.forEach((f, i) => {
        const item = document.createElement('div');
        item.className = 'ha-marker-item hit';
        item.innerHTML = `<span>Frame ${f}</span><button class="ha-del-btn">✕</button>`;
        item.querySelector('span').onclick = () => haSetFrame(f);
        item.querySelector('.ha-del-btn').onclick = (e) => {
            e.stopPropagation();
            haHits.splice(i, 1);
            haUpdateUI();
        };
        el.appendChild(item);
    });
}

function haRenderCropList() {
    const el = document.getElementById('ha-crop-list');
    el.innerHTML = '';
    if (haCrops.length === 0) {
        el.innerHTML = '<div style="font-size:0.75rem;color:#94a3b8">No cut points — video is one segment</div>';
        return;
    }

    // Show cut points with delete button
    haCrops.forEach((f, i) => {
        const item = document.createElement('div');
        item.className = 'ha-marker-item crop';
        item.innerHTML = `<span>C${i + 1}: Frame ${f}</span><button class="ha-del-btn">✕</button>`;
        item.querySelector('span').onclick = () => haSetFrame(f);
        item.querySelector('.ha-del-btn').onclick = (e) => {
            e.stopPropagation();
            haCrops.splice(i, 1);
            // Re-map ignored segments: remove any that referenced deleted cut, shift indices above it
            const newIgnored = new Set();
            haIgnoredSegments.forEach(idx => {
                if (idx < i) newIgnored.add(idx);
                else if (idx > i) newIgnored.add(idx - 1);
                // idx === i (segment after deleted cut) is dropped
            });
            haIgnoredSegments = newIgnored;
            haUpdateUI();
        };
        el.appendChild(item);
    });

    // Show segments summary
    const boundaries = [0, ...haCrops, haTotalFrames];
    for (let i = 0; i < boundaries.length - 1; i++) {
        const ignored = haIgnoredSegments.has(i);
        const seg = document.createElement('div');
        seg.style.cssText = `font-size:0.72rem;padding:2px 4px;margin-top:2px;border-radius:4px;cursor:pointer;display:flex;align-items:center;gap:4px;background:${ignored ? 'rgba(239,68,68,0.12)' : 'rgba(219,234,254,0.5)'};`;
        seg.innerHTML = `<span style="flex:1;color:${ignored ? '#ef4444' : '#2563eb'}">Seg ${i + 1}: ${boundaries[i]}–${boundaries[i + 1]} ${ignored ? '[X ignored]' : ''}</span>`;
        seg.onclick = () => {
            // Seek to start of segment
            haSetFrame(boundaries[i]);
        };
        el.appendChild(seg);
    }
}

function haAddHit(frame) {
    if (haHits.includes(frame)) return;
    haSaveHistory();
    haHits.push(frame);
    haHits.sort((a, b) => a - b);
    haUpdateUI();
}

function haAddCrop(frame) {
    if (haCrops.includes(frame)) return;
    haSaveHistory();
    haCrops.push(frame);
    haCrops.sort((a, b) => a - b);
    haUpdateUI();
}

function haGetSegmentIndex(frame) {
    // Returns the segment index (0..haCrops.length) that contains the given frame
    for (let i = 0; i < haCrops.length; i++) {
        if (frame < haCrops[i]) return i;
    }
    return haCrops.length;
}

function haToggleIgnoreSegment(frame) {
    haSaveHistory();
    const idx = haGetSegmentIndex(frame);
    if (haIgnoredSegments.has(idx)) {
        haIgnoredSegments.delete(idx);
    } else {
        haIgnoredSegments.add(idx);
    }
    haUpdateUI();
}

function haDeleteNearestHit(frame) {
    let bestDist = Infinity, bestIdx = -1;
    haHits.forEach((f, i) => {
        const d = Math.abs(f - frame);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
    });
    if (bestIdx < 0) return;
    haSaveHistory();
    haHits.splice(bestIdx, 1);
    haUpdateUI();
}

async function haSave() {
    const workdir = document.getElementById('ha-workdir-input').value.trim();
    if (!workdir) { alert('Please set a work directory'); return; }
    if (!haVideoFile) { alert('No video loaded'); return; }

    // Build non-ignored segments
    const boundaries = [0, ...haCrops, haTotalFrames];
    const segments = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
        if (haIgnoredSegments.has(i)) continue;
        const start = boundaries[i], end = boundaries[i + 1];
        const hits = haHits.filter(h => h >= start && h < end).map(h => h - start);
        segments.push({ start, end, hits });
    }
    if (segments.length === 0) { alert('No active segments to save'); return; }

    const btn = document.getElementById('ha-save-btn');
    btn.disabled = true;
    btn.textContent = `Uploading ${segments.length} clip(s)...`;

    try {
        const form = new FormData();
        form.append('workdir', workdir);
        form.append('fps', String(haFps));
        form.append('segments_json', JSON.stringify(segments));
        form.append('video', haVideoFile, haVideoFileName);

        const res = await fetch('/api/hitannotator/save_clips', { method: 'POST', body: form });
        const data = await res.json();
        if (!data.ok) throw new Error(data.detail || 'Save failed');

        const first = data.saved[0].video.replace('.mp4', '');
        const last = data.saved[data.saved.length - 1].video.replace('.mp4', '');
        const label = data.saved.length === 1 ? first : `${first}–${last}`;
        btn.textContent = `Saved ${data.saved.length} clip(s) (${label})!`;
        setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 4000);
    } catch (e) {
        console.error('save:', e);
        alert('Save error: ' + e.message);
        btn.textContent = 'Save';
        btn.disabled = false;
    }
}

let haKeyHoldState = {
    ArrowLeft: { active: false, startTime: 0, lastRafTime: 0 },
    ArrowRight: { active: false, startTime: 0, lastRafTime: 0 }
};
let haKeyHoldRaf = null;

function haProcessKeyHold(time) {
    if (activeTab !== 'videoclipper') {
        haKeyHoldState.ArrowLeft.active = false;
        haKeyHoldState.ArrowRight.active = false;
        haKeyHoldRaf = null;
        return;
    }

    let handled = false;
    ['ArrowLeft', 'ArrowRight'].forEach(key => {
        let state = haKeyHoldState[key];
        if (state.active) {
            handled = true;
            let duration = time - state.startTime;

            // Wait 300ms before auto-repeating to simulate OS key repeat delay
            if (duration > 300) {
                // Throttle updates to ~30Hz (every 33ms)
                if (time - state.lastRafTime > 33) {
                    state.lastRafTime = time;
                    let step = 1;
                    if (duration > 1500) step = 15;
                    else if (duration > 800) step = 5;
                    else if (duration > 400) step = 2;

                    let dir = key === 'ArrowLeft' ? -1 : 1;
                    let nextFrame = haCurrentFrame + (dir * step);
                    nextFrame = Math.max(0, Math.min(nextFrame, haTotalFrames));

                    if (nextFrame !== haCurrentFrame) {
                        haCurrentFrame = nextFrame;
                        haSeekTo(nextFrame);
                        haUpdateFrameUI();
                    }
                }
            }
        }
    });

    if (handled) {
        haKeyHoldRaf = requestAnimationFrame(haProcessKeyHold);
    } else {
        haKeyHoldRaf = null;
    }
}

// ── Hit Annotator Keyboard ──────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (activeTab !== 'videoclipper') return;
    if (e.target.tagName === 'INPUT') return;

    if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        haUndo();
        return;
    }

    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        if (e.repeat) return; // Prevent OS key-repeat lag; handled by RAF

        let state = haKeyHoldState[e.key];
        if (!state.active) {
            state.active = true;
            state.startTime = performance.now();
            state.lastRafTime = performance.now();

            // Immediate single step
            let dir = e.key === 'ArrowLeft' ? -1 : 1;
            haSetFrame(haCurrentFrame + dir);

            if (!haKeyHoldRaf) {
                haKeyHoldRaf = requestAnimationFrame(haProcessKeyHold);
            }
        }
    } else if (e.key === ' ') {
        e.preventDefault();
        haAddHit(haCurrentFrame);
    } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        haAddCrop(haCurrentFrame);
    } else if (e.key === 'x' || e.key === 'X') {
        e.preventDefault();
        haToggleIgnoreSegment(haCurrentFrame);
    } else if (e.key === 'Backspace') {
        e.preventDefault();
        haDeleteNearestHit(haCurrentFrame);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        haSetFrame(haCurrentFrame - 10);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        haSetFrame(haCurrentFrame + 10);
    }
});

document.addEventListener('keyup', e => {
    if (activeTab !== 'videoclipper') return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        haKeyHoldState[e.key].active = false;
    }
});

// Initialize when tab is activated
haInit();

// ============================================================================
// PERSON LABELER TAB
// ============================================================================

let plWorkdir = '/Users/harryzhang/git/tempopeak/datasets/v1/clips';
let plClips = [];           // [{name, has_detection, hits, p1_track, p2_track, hitters}]
let plCurrentIdx = -1;
let plCurrentFrame = 0;
let plTotalFrames = 0;
let plFps = 30;
let plVidW = 1920;          // video width from detection file
let plVidH = 1080;          // video height from detection file
let plHits = [];            // [frame, ...]
let plHitters = [];         // [0|1|2, ...] same length as plHits, 0=unmarked, 1=P1, 2=P2
let plKeyFrames = {};       // {frameIdx: {p1: detIdx|{x,y}|null, p2: detIdx|{x,y}|null}}
let plResolved = {};       // {frameIdx: {p1: {x1,y1,x2,y2}|null, p2: {...}|null}}
let plDetections = {};      // {frameIdx(int): [{id, x1, y1, x2, y2}]}
let plHistory = [];
let plDirty = false;        // Track unsaved changes
let plDetPollTimer = null;
let plFrameImg = null;
let plCanvas = null;
let plCtx = null;
let plRenderState = { offX: 0, offY: 0, scale: 1, canvasW: 0, canvasH: 0 };

let plKeyHoldState = {
    ArrowLeft: { active: false, startTime: 0, lastRafTime: 0 },
    ArrowRight: { active: false, startTime: 0, lastRafTime: 0 }
};
let plKeyHoldRaf = null;

// Draw box mode
let plDrawBoxMode = false;
let plDrawingBox = null; // {x1, y1, x2, y2} while drawing
let plDrawStart = null;  // {x, y} in video coords

// ── IoU helper ─────────────────────────────────────────────────
function iou(a, b) {
    if (!a || !b) return 0;
    const ix1 = Math.max(a.x1, b.x1);
    const iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2);
    const iy2 = Math.min(a.y2, b.y2);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 0.0001);
}

// ── Propagation ─────────────────────────────────────────────────
function plPropagate() {
    plResolved = {};

    // Get all keyframes sorted
    const keyFrameNums = Object.keys(plKeyFrames).map(Number).sort((a, b) => a - b);
    if (keyFrameNums.length === 0) return;

    // Get all frame numbers that have detections
    const allFrames = Object.keys(plDetections).map(Number).sort((a, b) => a - b);
    if (allFrames.length === 0) return;

    // Helper: get box from keyframe value (det index, manual dot, or manual box)
    function getKfBox(kfValue, frame) {
        if (kfValue == null || kfValue === 'skip') return null;
        if (typeof kfValue === 'number') {
            // Det index
            const dets = plDetections[frame];
            if (dets && dets[kfValue]) {
                return { x1: dets[kfValue].x1, y1: dets[kfValue].y1, x2: dets[kfValue].x2, y2: dets[kfValue].y2 };
            }
            return null;
        } else if (typeof kfValue === 'object') {
            if (kfValue.x1 !== undefined) {
                // Manual box: use directly
                return { x1: kfValue.x1, y1: kfValue.y1, x2: kfValue.x2, y2: kfValue.y2 };
            } else if (kfValue.x !== undefined) {
                // Manual dot: create virtual box
                const halfW = plVidW * 0.025, halfH = plVidH * 0.05;
                return { x1: kfValue.x - halfW, y1: kfValue.y - halfH, x2: kfValue.x + halfW, y2: kfValue.y + halfH };
            }
        }
        return null;
    }

    // For each role (p1, p2), propagate from keyframes
    for (const role of ['p1', 'p2']) {
        // Collect anchors: [{frame, box}]
        const anchors = [];
        for (const kf of keyFrameNums) {
            const kfData = plKeyFrames[kf];
            if (kfData && kfData[role] != null) {
                const box = getKfBox(kfData[role], kf);
                if (box) anchors.push({ frame: kf, box });
            }
        }
        if (anchors.length === 0) continue;

        // Initialize all frames with null
        for (const f of allFrames) {
            if (!plResolved[f]) plResolved[f] = { p1: null, p2: null };
        }

        // Forward propagation from each anchor
        for (const anchor of anchors) {
            let currentBox = anchor.box;
            const startIdx = allFrames.indexOf(anchor.frame);

            // Propagate forward
            for (let i = startIdx; i < allFrames.length; i++) {
                const f = allFrames[i];

                // Stop at next keyframe that has this role set (barrier)
                if (i > startIdx && plKeyFrames[f] && plKeyFrames[f][role] != null) break;

                const dets = plDetections[f];
                if (!dets || dets.length === 0) continue;

                let bestDet = null, bestIou = 0;
                for (const det of dets) {
                    const iouVal = iou(currentBox, det);
                    if (iouVal > bestIou) { bestIou = iouVal; bestDet = det; }
                }

                // Skip if already resolved by another anchor with higher priority
                if (i > startIdx && plResolved[f][role]) continue;

                if (bestIou > 0.3 && bestDet) {
                    // Check conflict: same box already assigned to other role
                    const otherRole = role === 'p1' ? 'p2' : 'p1';
                    if (plResolved[f][otherRole]) {
                        const otherIou = iou(plResolved[f][otherRole], bestDet);
                        if (otherIou > 0.8 && i !== startIdx) continue;
                    }

                    plResolved[f][role] = { x1: bestDet.x1, y1: bestDet.y1, x2: bestDet.x2, y2: bestDet.y2 };
                    currentBox = bestDet;
                } else if (i === startIdx) {
                    // Keyframe itself - must use it
                    plResolved[f][role] = { x1: currentBox.x1, y1: currentBox.y1, x2: currentBox.x2, y2: currentBox.y2 };
                }
            }

            // Propagate backward
            currentBox = anchor.box;
            for (let i = startIdx - 1; i >= 0; i--) {
                const f = allFrames[i];

                // Stop at previous keyframe that has this role set (barrier)
                if (plKeyFrames[f] && plKeyFrames[f][role] != null) break;

                const dets = plDetections[f];
                if (!dets || dets.length === 0) continue;

                if (plResolved[f][role]) continue;

                let bestDet = null, bestIou = 0;
                for (const det of dets) {
                    const iouVal = iou(currentBox, det);
                    if (iouVal > bestIou) { bestIou = iouVal; bestDet = det; }
                }

                if (bestIou > 0.3 && bestDet) {
                    const otherRole = role === 'p1' ? 'p2' : 'p1';
                    if (plResolved[f][otherRole] && iou(plResolved[f][otherRole], bestDet) > 0.8) continue;

                    plResolved[f][role] = { x1: bestDet.x1, y1: bestDet.y1, x2: bestDet.x2, y2: bestDet.y2 };
                    currentBox = bestDet;
                }
            }
        }
    }

}

// ── Seek ──────────────────────────────────────────────────────
function plSeekTo(frame) {
    if (plCurrentIdx < 0 || plTotalFrames <= 0) return;
    plCurrentFrame = Math.max(0, Math.min(frame, plTotalFrames - 1));
    document.getElementById('pl-frame-input').value = plCurrentFrame;
    const clip = plClips[plCurrentIdx];
    const url = `/api/pl/frame?workdir=${encodeURIComponent(plWorkdir)}&name=${clip.name}&frame=${plCurrentFrame}`;
    plFrameImg.onload = () => plDrawCanvas();
    plFrameImg.src = url;

    // Prefetch nearby frames
    for (let delta of [-1, 1, 2, -2]) {
        const pf = plCurrentFrame + delta;
        if (pf >= 0 && pf < plTotalFrames) {
            const img = new Image();
            img.src = `/api/pl/frame?workdir=${encodeURIComponent(plWorkdir)}&name=${clip.name}&frame=${pf}`;
        }
    }
}

// ── Canvas drawing ────────────────────────────────────────────
function plDrawCanvas() {
    if (!plCanvas || !plFrameImg || !plFrameImg.naturalWidth) {
        return;
    }
    const wrap = document.getElementById('pl-video-wrap');
    const canvasW = wrap.clientWidth, canvasH = wrap.clientHeight;
    // Use detection file dimensions for consistent scaling
    const vw = plVidW || plFrameImg.naturalWidth, vh = plVidH || plFrameImg.naturalHeight;
    if (!vw || !vh || !canvasW || !canvasH) {
        // Retry after layout settles (only if tab is active)
        if (activeTab === 'personlabel') {
            requestAnimationFrame(() => plDrawCanvas());
        }
        return;
    }

    const scale = Math.min(canvasW / vw, canvasH / vh);
    const renderW = vw * scale, renderH = vh * scale;
    const offX = (canvasW - renderW) / 2;
    const offY = (canvasH - renderH) / 2;
    plRenderState = { offX, offY, scale, canvasW, canvasH };

    const dpr = window.devicePixelRatio || 1;
    plCanvas.width = canvasW * dpr;
    plCanvas.height = canvasH * dpr;
    plCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    plCtx.clearRect(0, 0, canvasW, canvasH);

    // Find current hitter for HIT frame
    const hitIdx = plHits.indexOf(plCurrentFrame);
    const currentHitter = hitIdx >= 0 ? (plHitters[hitIdx] || 0) : 0;

    // Draw all detection boxes - ALWAYS show (gray if not assigned)
    const frameDets = plDetections[plCurrentFrame];
    const resolved = plResolved[plCurrentFrame] || {};
    const keyframe = plKeyFrames[plCurrentFrame];

    if (frameDets && frameDets.length > 0) {
        // Track which roles have already been drawn (prevent duplicate P1/P2 from overlapping detections)
        let p1Drawn = false, p2Drawn = false;

        frameDets.forEach((det, detIdx) => {
            const x1 = offX + det.x1 * scale, y1 = offY + det.y1 * scale;
            const x2 = offX + det.x2 * scale, y2 = offY + det.y2 * scale;

            // Determine box role based on resolved boxes
            let color = '#64748b'; // dark gray for unassigned candidates
            let label = `#${detIdx}`;
            let lineWidth = 1;
            let isHitter = false;
            let isKeyframe = false;

            // Only mark ONE det as P1, ONE as P2 (prevent duplicate highlights from overlapping detections)
            if (!p1Drawn && resolved.p1 && iou(det, resolved.p1) > 0.5) {
                color = '#10b981'; // green for P1
                label = 'P1';
                lineWidth = 2;
                if (currentHitter === 1) isHitter = true;
                if (keyframe && typeof keyframe.p1 === 'number' && keyframe.p1 === detIdx) isKeyframe = true;
                p1Drawn = true;
            }
            // Check if this detection matches resolved P2
            else if (!p2Drawn && resolved.p2 && iou(det, resolved.p2) > 0.5) {
                color = '#3b82f6'; // blue for P2
                label = 'P2';
                lineWidth = 2;
                if (currentHitter === 2) isHitter = true;
                if (keyframe && typeof keyframe.p2 === 'number' && keyframe.p2 === detIdx) isKeyframe = true;
                p2Drawn = true;
            }

            // Draw box
            plCtx.strokeStyle = color;
            plCtx.lineWidth = isHitter ? 4 : lineWidth;
            plCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Glow effect for hitter
            if (isHitter) {
                plCtx.shadowColor = color;
                plCtx.shadowBlur = 10;
                plCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                plCtx.shadowBlur = 0;
            }

            // Label with star for keyframes
            const displayLabel = isKeyframe ? `${label} ★` : label;
            plCtx.font = 'bold 11px sans-serif';
            const tw = plCtx.measureText(displayLabel).width;
            plCtx.fillStyle = color;
            plCtx.fillRect(x1, y1 - 16, tw + 8, 16);
            plCtx.fillStyle = '#fff';
            plCtx.fillText(displayLabel, x1 + 4, y1 - 4);
        });
    }

    // Draw manual boxes (not dots) from keyframes
    if (keyframe) {
        const drawManualBox = (kfValue, color, label) => {
            if (!kfValue || typeof kfValue !== 'object' || kfValue.x1 === undefined) return;
            const x1 = offX + kfValue.x1 * scale, y1 = offY + kfValue.y1 * scale;
            const x2 = offX + kfValue.x2 * scale, y2 = offY + kfValue.y2 * scale;

            plCtx.strokeStyle = color;
            plCtx.lineWidth = 3;
            plCtx.setLineDash([5, 3]);
            plCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            plCtx.setLineDash([]);

            plCtx.font = 'bold 11px sans-serif';
            const displayLabel = `${label} ★`;
            const tw = plCtx.measureText(displayLabel).width;
            plCtx.fillStyle = color;
            plCtx.fillRect(x1, y1 - 16, tw + 8, 16);
            plCtx.fillStyle = '#fff';
            plCtx.fillText(displayLabel, x1 + 4, y1 - 4);
        };
        drawManualBox(keyframe.p1, '#10b981', 'P1');
        drawManualBox(keyframe.p2, '#3b82f6', 'P2');

        // Draw manual dots (if not a box)
        const drawDot = (pt, color, label, role) => {
            if (!pt || typeof pt !== 'object' || pt.x1 !== undefined) return; // Skip if it's a box
            const hasResolved = resolved && resolved[role];
            const cx = offX + pt.x * scale;
            const cy = offY + pt.y * scale;
            const radius = 8;

            plCtx.fillStyle = color;
            plCtx.beginPath();
            plCtx.arc(cx, cy, radius, 0, Math.PI * 2);
            plCtx.fill();

            plCtx.strokeStyle = '#fff';
            plCtx.lineWidth = 2;
            plCtx.stroke();

            const displayLabel = hasResolved ? `${label} ★` : label;
            plCtx.font = 'bold 10px sans-serif';
            plCtx.fillStyle = '#fff';
            plCtx.textAlign = 'center';
            plCtx.fillText(displayLabel, cx, cy + 4);
            plCtx.textAlign = 'left';
        };
        drawDot(keyframe.p1, '#10b981', 'P1', 'p1');
        drawDot(keyframe.p2, '#3b82f6', 'P2', 'p2');
    }

    // Draw the box being drawn
    if (plDrawingBox) {
        const x1 = offX + plDrawingBox.x1 * scale, y1 = offY + plDrawingBox.y1 * scale;
        const x2 = offX + plDrawingBox.x2 * scale, y2 = offY + plDrawingBox.y2 * scale;

        plCtx.strokeStyle = '#f59e0b';
        plCtx.lineWidth = 2;
        plCtx.setLineDash([5, 3]);
        plCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        plCtx.setLineDash([]);

        // Fill with transparent yellow
        plCtx.fillStyle = 'rgba(245, 158, 11, 0.1)';
        plCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }

    // HIT marker at top center
    const hitEl = document.getElementById('pl-hit-indicator');
    if (hitIdx >= 0) {
        let c = '#ef4444', txt = 'HIT';
        if (currentHitter === 1) { c = '#10b981'; txt = 'HIT - P1'; }
        else if (currentHitter === 2) { c = '#3b82f6'; txt = 'HIT - P2'; }

        plCtx.fillStyle = c;
        plCtx.beginPath();
        plCtx.moveTo(canvasW / 2, 6);
        plCtx.lineTo(canvasW / 2 - 10, 24);
        plCtx.lineTo(canvasW / 2 + 10, 24);
        plCtx.closePath();
        plCtx.fill();
        hitEl.textContent = txt;
        hitEl.style.color = c;
    } else {
        hitEl.textContent = '';
    }

    // Update person bar - count keyframes
    let p1KfCount = 0, p2KfCount = 0;
    for (const kf of Object.values(plKeyFrames)) {
        if (kf.p1 != null) p1KfCount++;
        if (kf.p2 != null) p2KfCount++;
    }
    document.getElementById('pl-p1-label').textContent = `P1: ${p1KfCount} ★`;
    document.getElementById('pl-p2-label').textContent = `P2: ${p2KfCount} ★`;

    // Count annotated HITs
    const annotated = plHitters.filter(h => h !== 0).length;
    document.getElementById('pl-assign-info').textContent = `${annotated}/${plHits.length} hitters`;

    // Update missing stats
    plUpdateMissingStats();
}

// ── Frame navigation ──────────────────────────────────────────
function plSetFrame(frame) {
    frame = Math.max(0, Math.min(frame, plTotalFrames - 1));
    plCurrentFrame = frame;
    document.getElementById('pl-frame-input').value = frame;
    plSeekTo(frame);
}

function plNextHit(dir) {
    if (plHits.length === 0) { plSetFrame(plCurrentFrame + dir); return; }
    if (dir > 0) {
        const next = plHits.find(h => h > plCurrentFrame);
        plSetFrame(next !== undefined ? next : plHits[0]);
    } else {
        const prev = [...plHits].reverse().find(h => h < plCurrentFrame);
        plSetFrame(prev !== undefined ? prev : plHits[plHits.length - 1]);
    }
}

// ── Missing box stats and navigation ─────────────────────────────────────────────────
function plUpdateMissingStats() {
    const allFrames = Object.keys(plDetections).map(Number).sort((a, b) => a - b);
    let missingP1 = 0, missingP2 = 0;
    for (const f of allFrames) {
        const res = plResolved[f] || {};
        if (!res.p1) missingP1++;
        if (!res.p2) missingP2++;
    }
    document.getElementById('pl-stats-missing').textContent = `Missing: P1=${missingP1}, P2=${missingP2}`;
    return { missingP1, missingP2 };
}

function plJumpToMissing(role) {
    const allFrames = Object.keys(plDetections).map(Number).sort((a, b) => a - b);
    // Find next frame missing this role (after current frame)
    let nextMissing = allFrames.find(f => f > plCurrentFrame && !(plResolved[f] && plResolved[f][role]));
    if (nextMissing === undefined) {
        // Wrap around to beginning
        nextMissing = allFrames.find(f => !(plResolved[f] && plResolved[f][role]));
    }
    if (nextMissing !== undefined) {
        plSetFrame(nextMissing);
    }
}

// ── Save to memory (clip object) ─────────────────────────────────────────────────
function plSaveToMemory() {
    if (plCurrentIdx < 0) return;
    const clip = plClips[plCurrentIdx];
    clip.hits = [...plHits];
    clip.key_frames = JSON.parse(JSON.stringify(plKeyFrames));
    clip.hitters = [...plHitters];
    clip.resolved = JSON.parse(JSON.stringify(plResolved));
}

// ── Clip list ─────────────────────────────────────────────────
async function plLoadClips() {
    const workdir = document.getElementById('pl-workdir-input').value.trim();
    if (!workdir) return;
    plWorkdir = workdir;
    const res = await fetch(`/api/pl/clips?workdir=${encodeURIComponent(workdir)}`);
    const data = await res.json();
    plClips = data.clips;
    plRenderClipList();
    if (plClips.length > 0 && plCurrentIdx < 0) plSelectClip(0);
}

function plRenderClipList() {
    const el = document.getElementById('pl-clip-list');
    el.innerHTML = '';
    plClips.forEach((clip, i) => {
        const item = document.createElement('div');
        // Count keyframes and hitters
        const keyFrames = clip.key_frames || {};
        let kfP1 = 0, kfP2 = 0;
        for (const kf of Object.values(keyFrames)) {
            if (kf.p1 != null) kfP1++;
            if (kf.p2 != null) kfP2++;
        }
        const hitters = clip.hitters || [];
        const annotated = hitters.filter(h => h !== 0).length;
        const total = (clip.hits || []).length;
        const hasKeyframes = kfP1 > 0 || kfP2 > 0;
        item.className = 'clip-list-item' +
            (annotated > 0 || hasKeyframes ? ' annotated' : '') +
            (i === plCurrentIdx ? ' active' : '');
        const det = clip.has_detection ? '✓' : '·';
        const kfStatus = (kfP1 > 0 ? `P1:${kfP1}` : '') + (kfP2 > 0 ? (kfP1 > 0 ? `+P2:${kfP2}` : `P2:${kfP2}`) : '');
        item.textContent = `${clip.name} [${det}] ${kfStatus} ${annotated}/${total}`;
        item.onclick = () => plSelectClip(i);
        el.appendChild(item);
    });
}

async function plSelectClip(idx) {
    if (idx < 0 || idx >= plClips.length) return;

    // Auto-save current clip to file before switching
    if (plCurrentIdx >= 0 && plCurrentIdx !== idx && plDirty) {
        await plSave();
    } else if (plCurrentIdx >= 0 && plCurrentIdx !== idx) {
        plSaveToMemory();
    }

    plCurrentIdx = idx;
    const clip = plClips[idx];

    plCurrentFrame = 0;
    plHistory = [];
    plDirty = false; // Reset dirty flag when loading new clip
    plHits = clip.hits || [];
    plKeyFrames = clip.key_frames || {};
    plHitters = clip.hitters || new Array(plHits.length).fill(0);

    // Load detections (includes fps, width, height)
    const detRes = await fetch(`/api/pl/detections?workdir=${encodeURIComponent(plWorkdir)}&name=${clip.name}`);
    const detData = await detRes.json();
    plFps = detData.fps || 30;
    plVidW = detData.width || 1920;
    plVidH = detData.height || 1080;
    plDetections = {};
    Object.entries(detData.frames || {}).forEach(([k, v]) => { plDetections[parseInt(k)] = v; });

    // Calculate total frames from detection data
    const frameKeys = Object.keys(plDetections).map(Number);
    plTotalFrames = frameKeys.length > 0 ? Math.max(...frameKeys) + 1 : 0;
    document.getElementById('pl-total-frames').textContent = plTotalFrames;

    // Load resolved from clip (either from memory or from JSON frames)
    if (clip.resolved) {
        // From memory (after switching between clips in same session)
        plResolved = JSON.parse(JSON.stringify(clip.resolved));
    } else if (clip.frames && Object.keys(clip.frames).length > 0) {
        // From saved JSON file (frames -> resolved)
        plResolved = {};
        for (const [frameStr, frameData] of Object.entries(clip.frames)) {
            plResolved[parseInt(frameStr)] = {
                p1: frameData.p1 || null,
                p2: frameData.p2 || null
            };
        }
    } else {
        // No saved state, start empty
        plResolved = {};
    }

    // Load first frame
    plSeekTo(0);
    plRenderClipList();
}

// ── Draw Box Mode ──────────────────────────────────────────────────────
function plToggleDrawBox() {
    plDrawBoxMode = !plDrawBoxMode;
    const btn = document.getElementById('pl-drawbox-btn');
    if (plDrawBoxMode) {
        btn.classList.add('active');
        btn.style.background = '#3b82f6';
        btn.style.color = '#fff';
        plCanvas.style.cursor = 'crosshair';
    } else {
        btn.classList.remove('active');
        btn.style.background = '';
        btn.style.color = '';
        plCanvas.style.cursor = '';
        plDrawingBox = null;
        plDrawStart = null;
        plDrawCanvas();
    }
}

function plHandleMouseDown(e) {
    if (!plDrawBoxMode) return;
    const rect = plCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const { offX, offY, scale } = plRenderState;
    const vx = (cx - offX) / scale;
    const vy = (cy - offY) / scale;
    plDrawStart = { x: vx, y: vy };
    plDrawingBox = { x1: vx, y1: vy, x2: vx, y2: vy };
}

function plHandleMouseMove(e) {
    if (!plDrawBoxMode || !plDrawStart) return;
    const rect = plCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const { offX, offY, scale } = plRenderState;
    const vx = (cx - offX) / scale;
    const vy = (cy - offY) / scale;
    plDrawingBox = {
        x1: Math.min(plDrawStart.x, vx),
        y1: Math.min(plDrawStart.y, vy),
        x2: Math.max(plDrawStart.x, vx),
        y2: Math.max(plDrawStart.y, vy)
    };
    plDrawCanvas();
}

function plHandleMouseUp(e) {
    if (!plDrawBoxMode || !plDrawingBox) return;

    let box = plDrawingBox;
    const minSize = 20; // Minimum box size

    // Clip box to video boundaries
    box = {
        x1: Math.max(0, Math.min(box.x1, plVidW - 1)),
        y1: Math.max(0, Math.min(box.y1, plVidH - 1)),
        x2: Math.max(1, Math.min(box.x2, plVidW)),
        y2: Math.max(1, Math.min(box.y2, plVidH))
    };

    if (box.x2 - box.x1 < minSize || box.y2 - box.y1 < minSize) {
        // Too small, cancel
        plDrawingBox = null;
        plDrawStart = null;
        plDrawCanvas();
        return;
    }

    plSaveHistory();

    // Check if this box overlaps with existing DOT
    let kf = plKeyFrames[plCurrentFrame] || { p1: null, p2: null };
    const boxCenter = { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 };
    const DOT_RADIUS = Math.max(30, plVidW * 0.02);

    let assigned = false;
    // Check P1 DOT
    if (typeof kf.p1 === 'object' && kf.p1) {
        const dist = Math.hypot(boxCenter.x - kf.p1.x, boxCenter.y - kf.p1.y);
        if (dist < DOT_RADIUS) {
            // Replace P1 DOT with box
            kf.p1 = box;
            assigned = true;
        }
    }
    // Check P2 DOT
    if (!assigned && typeof kf.p2 === 'object' && kf.p2) {
        const dist = Math.hypot(boxCenter.x - kf.p2.x, boxCenter.y - kf.p2.y);
        if (dist < DOT_RADIUS) {
            // Replace P2 DOT with box
            kf.p2 = box;
            assigned = true;
        }
    }

    // If no DOT to replace, assign to whichever role is free
    // Check BOTH keyframe AND resolved (propagated) state
    if (!assigned) {
        const resolved = plResolved[plCurrentFrame] || { p1: null, p2: null };
        const hasP1 = (kf.p1 != null && kf.p1 !== 'skip') || resolved.p1 != null;
        const hasP2 = (kf.p2 != null && kf.p2 !== 'skip') || resolved.p2 != null;

        if (!hasP1 && !hasP2) {
            kf.p1 = box;
        } else if (hasP1 && !hasP2) {
            kf.p2 = box;
        } else if (!hasP1 && hasP2) {
            kf.p1 = box;
        }
        // If both taken, don't overwrite (user needs to remove first)
    }

    plKeyFrames[plCurrentFrame] = kf;

    // Clear draw state
    plDrawingBox = null;
    plDrawStart = null;

    // Propagate and redraw
    plPropagate();
    plDrawCanvas();

    // Auto-mark hitter if on HIT frame
    const hitIdx = plHits.indexOf(plCurrentFrame);
    if (hitIdx >= 0 && plHitters[hitIdx] === 0) {
        if (kf.p1 && typeof kf.p1 === 'object' && kf.p1.x1 !== undefined) {
            plHitters[hitIdx] = 1;
        } else if (kf.p2 && typeof kf.p2 === 'object' && kf.p2.x1 !== undefined) {
            plHitters[hitIdx] = 2;
        }
    }
}

// ── Canvas click: assign P1/P2 keyframe or mark hitter ────────────────────────────────
function plHandleCanvasClick(e) {
    // Don't handle click if in draw box mode (mouse handlers will take over)
    if (plDrawBoxMode) return;

    const rect = plCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const { offX, offY, scale } = plRenderState;

    // Convert canvas coords → video coords
    const vx = (cx - offX) / scale;
    const vy = (cy - offY) / scale;

    // Find clicked detection box
    const frameDets = plDetections[plCurrentFrame];
    let clickedIdx = -1;
    if (frameDets && frameDets.length > 0) {
        for (let i = 0; i < frameDets.length; i++) {
            const det = frameDets[i];
            if (vx >= det.x1 && vx <= det.x2 && vy >= det.y1 && vy <= det.y2) {
                clickedIdx = i;
                break;
            }
        }
    }

    plSaveHistory();

    const hitIdx = plHits.indexOf(plCurrentFrame);
    let kf = plKeyFrames[plCurrentFrame] || { p1: null, p2: null };
    const resolved = plResolved[plCurrentFrame] || {};

    // --- Determine clicked entity and its current role ---
    let clickedRole = null;  // 'p1', 'p2', or null (unassigned)
    let kfValue = null;
    let clickedEntity = null; // 'detection', 'box', 'dot', or null
    const isHitFrame = hitIdx >= 0;

    // 1. Detection box
    if (clickedIdx >= 0) {
        clickedEntity = 'detection';
        kfValue = clickedIdx;
        const clickedDet = frameDets[clickedIdx];
        const isP1 = (kf.p1 === clickedIdx) || (resolved.p1 && iou(clickedDet, resolved.p1) > 0.5);
        const isP2 = (kf.p2 === clickedIdx) || (resolved.p2 && iou(clickedDet, resolved.p2) > 0.5);
        if (isP1) clickedRole = 'p1';
        else if (isP2) clickedRole = 'p2';
    }

    // 2. Drawn keyframe box (x1/y1/x2/y2)
    if (!clickedEntity) {
        const p1IsBox = kf.p1 && typeof kf.p1 === 'object' && kf.p1.x1 !== undefined;
        const p2IsBox = kf.p2 && typeof kf.p2 === 'object' && kf.p2.x1 !== undefined;
        if (p1IsBox && vx >= kf.p1.x1 && vx <= kf.p1.x2 && vy >= kf.p1.y1 && vy <= kf.p1.y2) {
            clickedEntity = 'box';
            clickedRole = 'p1';
            kfValue = kf.p1;
        } else if (p2IsBox && vx >= kf.p2.x1 && vx <= kf.p2.x2 && vy >= kf.p2.y1 && vy <= kf.p2.y2) {
            clickedEntity = 'box';
            clickedRole = 'p2';
            kfValue = kf.p2;
        }
    }

    // 3. DOT ({x, y})
    if (!clickedEntity) {
        const DOT_R = Math.max(30, plVidW * 0.02);
        const p1IsDot = kf.p1 && typeof kf.p1 === 'object' && kf.p1.x !== undefined && kf.p1.x1 === undefined;
        const p2IsDot = kf.p2 && typeof kf.p2 === 'object' && kf.p2.x !== undefined && kf.p2.x1 === undefined;
        if (p1IsDot && Math.hypot(vx - kf.p1.x, vy - kf.p1.y) < DOT_R) {
            clickedEntity = 'dot';
            clickedRole = 'p1';
            kfValue = kf.p1;
        } else if (p2IsDot && Math.hypot(vx - kf.p2.x, vy - kf.p2.y) < DOT_R) {
            clickedEntity = 'dot';
            clickedRole = 'p2';
            kfValue = kf.p2;
        }
    }

    // --- Unified cycle: P1 → P1 HIT → P2 → P2 HIT → discard ---
    if (clickedEntity) {
        if (clickedRole === null) {
            // Unassigned detection → assign to first available role
            const p1Free = (kf.p1 == null || kf.p1 === 'skip') && resolved.p1 == null;
            const p2Free = (kf.p2 == null || kf.p2 === 'skip') && resolved.p2 == null;
            if (p1Free) kf.p1 = kfValue;
            else if (p2Free) kf.p2 = kfValue;
            plKeyFrames[plCurrentFrame] = kf;
        } else {
            const hitterNum = clickedRole === 'p1' ? 1 : 2;
            const isHitter = isHitFrame && plHitters[hitIdx] === hitterNum;
            const otherRole = clickedRole === 'p1' ? 'p2' : 'p1';
            const otherTaken = (kf[otherRole] != null && kf[otherRole] !== 'skip') || resolved[otherRole] != null;

            if (!isHitter && isHitFrame) {
                // P → P HIT (mark as hitter)
                plHitters[hitIdx] = hitterNum;
                if (kf[clickedRole] == null || kf[clickedRole] === 'skip') {
                    kf[clickedRole] = kfValue;
                    plKeyFrames[plCurrentFrame] = kf;
                }
            } else if (clickedRole === 'p1' && !otherTaken) {
                // P1 (HIT or non-HIT) → P2
                const val = (kf.p1 != null && kf.p1 !== 'skip') ? kf.p1 : kfValue;
                kf.p2 = val;
                delete kf.p1;  // delete (not skip) so P1 propagation passes through
                plKeyFrames[plCurrentFrame] = kf;
                if (isHitFrame) plHitters[hitIdx] = 0;
            } else {
                // → discard (use 'skip' sentinel to block propagation)
                kf[clickedRole] = 'skip';
                plKeyFrames[plCurrentFrame] = kf;
                if (isHitFrame) plHitters[hitIdx] = 0;
            }
        }

        plPropagate();
        plDrawCanvas();
        return;
    }

    // --- Empty canvas: place new DOT ---
    const p1Taken = (kf.p1 != null && kf.p1 !== 'skip') || resolved.p1 != null;
    const p2Taken = (kf.p2 != null && kf.p2 !== 'skip') || resolved.p2 != null;
    let placedRole = null;

    if (!p1Taken) {
        kf.p1 = { x: vx, y: vy };
        placedRole = 'p1';
    } else if (!p2Taken) {
        kf.p2 = { x: vx, y: vy };
        placedRole = 'p2';
    }

    if (placedRole) {
        plKeyFrames[plCurrentFrame] = kf;
        plResolveDot(plCurrentFrame, placedRole);
    }

    // Re-propagate and redraw
    plPropagate();
    plDrawCanvas();
}

// ── Undo ──────────────────────────────────────────────────────
function plSaveHistory() {
    plDirty = true; // Mark as having unsaved changes
    plHistory.push({
        keyFrames: JSON.parse(JSON.stringify(plKeyFrames)),
        hitters: [...plHitters],
        hits: [...plHits]
    });
    if (plHistory.length > 50) plHistory.shift();
}
function plUndo() {
    if (plHistory.length === 0) return;
    const state = plHistory.pop();
    plKeyFrames = state.keyFrames;
    plHitters = state.hitters;
    plHits = state.hits;
    plPropagate();
    plDrawCanvas();
    plUpdateHitIndicator();
}

// ── Toggle HIT on current frame ──────────────────────────────
function plToggleHit() {
    if (plCurrentIdx < 0) return;
    plSaveHistory();

    const hitIdx = plHits.indexOf(plCurrentFrame);
    if (hitIdx >= 0) {
        // Remove HIT
        plHits.splice(hitIdx, 1);
        plHitters.splice(hitIdx, 1);
    } else {
        // Add HIT
        // Keep sorted
        let insertIdx = 0;
        while (insertIdx < plHits.length && plHits[insertIdx] < plCurrentFrame) {
            insertIdx++;
        }
        plHits.splice(insertIdx, 0, plCurrentFrame);
        plHitters.splice(insertIdx, 0, 0); // Default hitter = 0 (unmarked)
    }

    plDrawCanvas();
    plUpdateHitIndicator();
}

function plUpdateHitIndicator() {
    const hitIdx = plHits.indexOf(plCurrentFrame);
    const el = document.getElementById('pl-hit-indicator');
    if (hitIdx >= 0) {
        const h = plHitters[hitIdx] || 0;
        let txt = 'HIT';
        if (h === 1) txt = 'HIT - P1';
        else if (h === 2) txt = 'HIT - P2';
        el.textContent = txt;
        el.style.color = h === 1 ? '#10b981' : h === 2 ? '#3b82f6' : '#ef4444';
    } else {
        el.textContent = '';
    }
}

// ── Helper: Clip box to video boundaries ──────────────────────
function clipBox(box, vidW, vidH) {
    if (!box) return null;
    return {
        x1: Math.max(0, Math.min(box.x1, vidW - 1)),
        y1: Math.max(0, Math.min(box.y1, vidH - 1)),
        x2: Math.max(1, Math.min(box.x2, vidW)),
        y2: Math.max(1, Math.min(box.y2, vidH))
    };
}

// ── Save ──────────────────────────────────────────────────────
async function plSave() {
    if (plCurrentIdx < 0) return;
    const clip = plClips[plCurrentIdx];
    const btn = document.getElementById('pl-save-btn');
    btn.textContent = 'Saving...'; btn.disabled = true;

    // Build frames object from resolved data (clip boxes to video bounds)
    const frames = {};
    for (const [frameStr, resolved] of Object.entries(plResolved)) {
        const frameObj = { p1: null, p2: null };
        if (resolved.p1) frameObj.p1 = clipBox(resolved.p1, plVidW, plVidH);
        if (resolved.p2) frameObj.p2 = clipBox(resolved.p2, plVidW, plVidH);
        if (frameObj.p1 || frameObj.p2) {
            frames[frameStr] = frameObj;
        }
    }

    try {
        const res = await fetch('/api/pl/save', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workdir: plWorkdir,
                name: clip.name,
                hits: plHits,
                hitters: plHitters,
                key_frames: plKeyFrames,
                frames
            })
        });
        const data = await res.json();
        if (data.ok) {
            // Update memory
            clip.key_frames = JSON.parse(JSON.stringify(plKeyFrames));
            clip.hitters = [...plHitters];
            clip.resolved = JSON.parse(JSON.stringify(plResolved));
                    plDirty = false; // Clear dirty flag after successful save
            plRenderClipList();
            btn.textContent = 'Saved!';
            setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 2000);
        } else { throw new Error(data.detail || 'Save failed'); }
    } catch (e) {
        alert('Save error: ' + e.message);
        btn.textContent = 'Save'; btn.disabled = false;
    }
}

// ── Export All (with modal progress) ──────────────────────────────
let plExportPollTimer = null;

function plShowExportOverlay(show) {
    let overlay = document.getElementById('pl-export-overlay');
    if (!overlay && show) {
        overlay = document.createElement('div');
        overlay.id = 'pl-export-overlay';
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;' +
            'background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:9999;';
        overlay.innerHTML = `
            <div style="background:#1e293b;border-radius:12px;padding:32px 48px;min-width:360px;text-align:center;color:#e2e8f0;">
                <div style="font-size:1.2rem;font-weight:600;margin-bottom:16px;">Exporting...</div>
                <div id="pl-export-bar-wrap" style="background:#334155;border-radius:6px;height:20px;overflow:hidden;margin-bottom:12px;">
                    <div id="pl-export-bar" style="background:#8b5cf6;height:100%;width:0%;transition:width 0.3s;border-radius:6px;"></div>
                </div>
                <div id="pl-export-text" style="font-size:0.85rem;color:#94a3b8;">Preparing...</div>
            </div>`;
        document.body.appendChild(overlay);
    }
    if (overlay) overlay.style.display = show ? 'flex' : 'none';
}

async function plExport() {
    if (!plWorkdir) return;

    // Save current clip first
    if (plCurrentIdx >= 0 && plDirty) {
        await plSave();
    }

    // Let user pick export directory
    let exportDir;
    try {
        const browseRes = await fetch('/api/browse_dir');
        const browseData = await browseRes.json();
        if (!browseData.path) return; // cancelled
        exportDir = browseData.path;
    } catch (e) {
        alert('Browse failed'); return;
    }

    // Start export on server
    const res = await fetch('/api/pl/export_start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workdir: plWorkdir, export_dir: exportDir })
    });
    const data = await res.json();
    if (!data.ok) { alert('Export failed: ' + (data.detail || '')); return; }

    // Show overlay and start polling
    plShowExportOverlay(true);
    plExportPollTimer = setInterval(plPollExportStatus, 500);
}

async function plPollExportStatus() {
    const s = await (await fetch('/api/pl/export_status')).json();
    const bar = document.getElementById('pl-export-bar');
    const text = document.getElementById('pl-export-text');
    if (!bar || !text) return;

    const pct = s.total_clips > 0 ? Math.round(s.done_clips / s.total_clips * 100) : 0;
    bar.style.width = pct + '%';

    if (s.running) {
        text.textContent = `${s.done_clips}/${s.total_clips} clips · ${s.done_frames} frames${s.current ? ' · ' + s.current : ''}`;
    } else {
        clearInterval(plExportPollTimer);
        plExportPollTimer = null;
        if (s.error) {
            text.textContent = `Error: ${s.error}`;
            bar.style.background = '#ef4444';
        } else {
            bar.style.width = '100%';
            text.textContent = `Done! ${s.done_clips} clips, ${s.done_frames} frames exported`;
        }
        // Auto-close after 2s
        setTimeout(() => plShowExportOverlay(false), 2000);
    }
}

// ── Clear Clip ──────────────────────────────────────────────────────
async function plClearClip() {
    if (plCurrentIdx < 0) return;

    const clearAll = document.getElementById('pl-clear-all').checked;
    const msg = clearAll
        ? 'Clear annotations for ALL clips? (HIT frames will be kept)'
        : 'Clear annotations for this clip? (HIT frames will be kept)';

    if (!confirm(msg)) return;

    if (clearAll) {
        // Clear all clips
        for (let i = 0; i < plClips.length; i++) {
            const clip = plClips[i];
            clip.key_frames = {};
            clip.hitters = (clip.hits || []).map(() => 0);
            clip.resolved = {};
            // Save to file
            await fetch('/api/pl/save', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    workdir: plWorkdir,
                    name: clip.name,
                    hits: clip.hits || [],
                    hitters: clip.hitters,
                    key_frames: clip.key_frames,
                    frames: {}
                })
            });
        }

        // Reset current state
        plKeyFrames = {};
        plHitters = plHits.map(() => 0);
        plResolved = {};
    } else {
        // Clear current clip only
        plSaveHistory();
        plKeyFrames = {};
        plHitters = plHits.map(() => 0);
        plResolved = {};

        // Update memory
        plSaveToMemory();

        // Save to file
        await plSave();
    }

    plDrawCanvas();
    plUpdateHitIndicator();
    plRenderClipList();
}

// ── Delete Clip ──────────────────────────────────────────────────────
async function plDeleteClip() {
    if (plCurrentIdx < 0) return;
    const clip = plClips[plCurrentIdx];
    if (!confirm(`Delete clip "${clip.name}"? The last clip will fill the gap.`)) return;

    // 1. Save current clip first (preserve any unsaved work on OTHER clips)
    await plSave();

    // 2. Delete on server
    const res = await fetch('/api/pl/delete_clip', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workdir: plWorkdir, name: clip.name })
    });
    const data = await res.json();
    if (!data.ok) { alert('Delete failed: ' + (data.detail || '')); return; }

    // 3. Clear in-memory state for all clips (force fresh load from disk)
    const prevIdx = plCurrentIdx;
    plCurrentIdx = -1;
    plClips = [];
    plKeyFrames = {};
    plHitters = [];
    plResolved = {};
    plDetections = {};
    plHistory = [];
    plDirty = false;

    // 4. Reload everything from disk
    await plLoadClips();
    if (plClips.length > 0) {
        await plSelectClip(Math.min(prevIdx, plClips.length - 1));
    }
}

// ── Fill Gaps ──────────────────────────────────────────────────────
async function plFillGaps() {
    if (plCurrentIdx < 0) return;
    const clip = plClips[plCurrentIdx];
    const btn = document.getElementById('pl-fillgaps-btn');
    btn.textContent = 'Scanning...';
    btn.disabled = true;

    // Check if we have detection data
    const allFrames = Object.keys(plDetections).map(Number).sort((a, b) => a - b);
    if (allFrames.length === 0) {
        btn.textContent = 'No detections';
        setTimeout(() => { btn.textContent = 'Fill Gaps'; btn.disabled = false; }, 1500);
        return;
    }

    // Find gap frames where P1 or P2 is missing in plResolved
    const gaps = [];
    for (const f of allFrames) {
        const resolved = plResolved[f] || { p1: null, p2: null };
        if (!resolved.p1 || !resolved.p2) {
            gaps.push({ frame: f, missingP1: !resolved.p1, missingP2: !resolved.p2 });
        }
    }

    if (gaps.length === 0) {
        btn.textContent = 'No gaps';
        setTimeout(() => { btn.textContent = 'Fill Gaps'; btn.disabled = false; }, 1500);
        return;
    }

    plSaveHistory();
    let filled = 0;
    let skipped = 0;

    // Helper: find nearest boxes on BOTH sides (sandwich method)
    const findSandwichBoxes = (role, gapFrame) => {
        let prevBox = null, prevFrame = null;
        let nextBox = null, nextFrame = null;

        // Search backward for previous box
        for (let i = allFrames.indexOf(gapFrame) - 1; i >= 0; i--) {
            const f = allFrames[i];
            const res = plResolved[f];
            if (res && res[role]) {
                prevBox = res[role];
                prevFrame = f;
                break;
            }
        }

        // Search forward for next box
        for (let i = allFrames.indexOf(gapFrame) + 1; i < allFrames.length; i++) {
            const f = allFrames[i];
            const res = plResolved[f];
            if (res && res[role]) {
                nextBox = res[role];
                nextFrame = f;
                break;
            }
        }

        return { prevBox, prevFrame, nextBox, nextFrame };
    };

    // Helper: create search box from two boxes (average center + 2x max length)
    const createSearchBox = (box1, box2) => {
        // Average center
        const cx = ((box1.x1 + box1.x2) / 2 + (box2.x1 + box2.x2) / 2) / 2;
        const cy = ((box1.y1 + box1.y2) / 2 + (box2.y1 + box2.y2) / 2) / 2;

        // Max length (width or height) from both boxes
        const w1 = box1.x2 - box1.x1, h1 = box1.y2 - box1.y1;
        const w2 = box2.x2 - box2.x1, h2 = box2.y2 - box2.y1;
        const maxLen = Math.max(w1, h1, w2, h2);
        const halfSize = maxLen; // 2x max length = 2 * maxLen, so half is maxLen

        return {
            x1: Math.max(0, Math.round(cx - halfSize)),
            y1: Math.max(0, Math.round(cy - halfSize)),
            x2: Math.min(plVidW, Math.round(cx + halfSize)),
            y2: Math.min(plVidH, Math.round(cy + halfSize))
        };
    };

    for (let i = 0; i < gaps.length; i++) {
        const gap = gaps[i];
        btn.textContent = `Filling... ${i + 1}/${gaps.length}`;

        const fillGap = async (role) => {
            const { prevBox, prevFrame, nextBox, nextFrame } = findSandwichBoxes(role, gap.frame);

            // REQUIRE both sides (sandwich method)
            if (!prevBox || !nextBox) {
                return false; // Can't fill without sandwich
            }

            // Create search box from average center + 2x max length
            const searchBox = createSearchBox(prevBox, nextBox);

            // Try YOLO detection on this frame
            try {
                const res = await fetch('/api/pl/fill_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        workdir: plWorkdir,
                        name: clip.name,
                        frame: gap.frame,
                        search_box: searchBox
                    })
                });
                const data = await res.json();
                if (data.box) {
                    if (!plResolved[gap.frame]) plResolved[gap.frame] = { p1: null, p2: null };
                    plResolved[gap.frame][role] = data.box;
                    return true;
                }
            } catch (e) {
                console.error('Fill gap error:', e);
            }

            // If YOLO failed, skip (don't interpolate)
            return false;
        };

        if (gap.missingP1) {
            if (await fillGap('p1')) filled++;
            else skipped++;
        }
        if (gap.missingP2) {
            if (await fillGap('p2')) filled++;
            else skipped++;
        }
    }

    plDrawCanvas();
    btn.textContent = `Filled ${filled}, skipped ${skipped}`;
    setTimeout(() => { btn.textContent = 'Fill Gaps'; btn.disabled = false; }, 3000);
}

// ── Resolve DOT to BOX via YOLO ───────────────────────────────────────
function findNearestResolvedBox(role, targetFrame) {
    for (let d = 1; d < 30; d++) {
        for (const f of [targetFrame - d, targetFrame + d]) {
            const res = plResolved[f];
            if (res && res[role]) return res[role];
        }
    }
    return null;
}

function findNearestResolvedBoxWithFrame(role, targetFrame) {
    let prevBox = null, prevFrame = null, nextBox = null, nextFrame = null;
    const allFrames = Object.keys(plResolved).map(Number).sort((a, b) => a - b);
    for (const f of allFrames) {
        if (plResolved[f][role]) {
            if (f < targetFrame) { prevBox = plResolved[f][role]; prevFrame = f; }
            else if (f > targetFrame && !nextBox) { nextBox = plResolved[f][role]; nextFrame = f; }
        }
    }
    return { prevBox, prevFrame, nextBox, nextFrame };
}

async function plResolveDot(frame, role) {
    const kf = plKeyFrames[frame];
    if (!kf || !kf[role] || typeof kf[role] !== 'object') return;
    const dot = kf[role];
    const clip = plClips[plCurrentIdx];
    if (!clip) return;

    // Build search box centered on dot
    const nearbyBox = findNearestResolvedBox(role, frame);
    let halfW, halfH;
    if (nearbyBox) {
        halfW = (nearbyBox.x2 - nearbyBox.x1) * 0.75;
        halfH = (nearbyBox.y2 - nearbyBox.y1) * 0.75;
    } else {
        halfW = plVidW * 0.05;
        halfH = plVidH * 0.1;
    }
    const searchBox = {
        x1: dot.x - halfW, y1: dot.y - halfH,
        x2: dot.x + halfW, y2: dot.y + halfH
    };

    try {
        const res = await fetch('/api/pl/fill_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workdir: plWorkdir, name: clip.name,
                frame: frame, search_box: searchBox
            })
        });
        const data = await res.json();
        if (data.box) {
            if (!plResolved[frame]) plResolved[frame] = { p1: null, p2: null };
            plResolved[frame][role] = data.box;
            plDrawCanvas();
            return;
        }
    } catch (e) { /* ignore */ }

    // YOLO failed: try interpolation
    const { prevBox, prevFrame, nextBox, nextFrame } = findNearestResolvedBoxWithFrame(role, frame);
    if (prevBox && nextBox && prevFrame != null && nextFrame != null) {
        const t = (frame - prevFrame) / (nextFrame - prevFrame);
        const interpBox = {
            x1: Math.round(prevBox.x1 + (nextBox.x1 - prevBox.x1) * t),
            y1: Math.round(prevBox.y1 + (nextBox.y1 - prevBox.y1) * t),
            x2: Math.round(prevBox.x2 + (nextBox.x2 - prevBox.x2) * t),
            y2: Math.round(prevBox.y2 + (nextBox.y2 - prevBox.y2) * t)
        };
        if (!plResolved[frame]) plResolved[frame] = { p1: null, p2: null };
        plResolved[frame][role] = interpBox;
    } else if (prevBox || nextBox) {
        if (!plResolved[frame]) plResolved[frame] = { p1: null, p2: null };
        plResolved[frame][role] = prevBox || nextBox;
    }

    plDrawCanvas();
}

// ── Detection ─────────────────────────────────────────────────
async function plStartDetection() {
    const workdir = document.getElementById('pl-workdir-input').value.trim();
    if (!workdir) return;

    const scope = document.getElementById('pl-detect-scope').value;
    const currentClip = plCurrentIdx >= 0 ? plClips[plCurrentIdx]?.name : null;

    if (scope === 'current' && !currentClip) {
        alert('No clip selected');
        return;
    }

    const btn = document.getElementById('pl-detect-btn');
    btn.disabled = true;
    const res = await fetch('/api/pl/detect_start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workdir, scope, current_clip: currentClip })
    });
    const data = await res.json();
    if (data.ok) {
        document.getElementById('pl-det-progress').textContent = 'Starting...';
        plDetPollTimer = setInterval(plPollDetStatus, 2000);
    } else {
        alert('Failed: ' + (data.detail || ''));
        btn.disabled = false;
    }
}

async function plPollDetStatus() {
    const s = await (await fetch('/api/pl/detect_status')).json();
    const el = document.getElementById('pl-det-progress');
    if (s.running) {
        el.textContent = `Detecting... ${s.done}/${s.total}${s.current ? ' (' + s.current + ')' : ''}`;
    } else {
        el.textContent = `Done: ${s.done}/${s.total} clips`;
        clearInterval(plDetPollTimer); plDetPollTimer = null;
        document.getElementById('pl-detect-btn').disabled = false;
        plLoadClips();
    }
}

// ── RAF key hold (same pattern as Hit Annotator) ──────────────
function plProcessKeyHold(time) {
    if (activeTab !== 'personlabel') {
        plKeyHoldState.ArrowLeft.active = false;
        plKeyHoldState.ArrowRight.active = false;
        plKeyHoldRaf = null; return;
    }
    let handled = false;
    ['ArrowLeft', 'ArrowRight'].forEach(key => {
        const state = plKeyHoldState[key];
        if (state.active) {
            handled = true;
            const duration = time - state.startTime;
            if (duration > 300 && time - state.lastRafTime > 33) {
                state.lastRafTime = time;
                let step = 1;
                if (duration > 1500) step = 15;
                else if (duration > 800) step = 5;
                else if (duration > 400) step = 2;
                plSetFrame(plCurrentFrame + (key === 'ArrowLeft' ? -1 : 1) * step);
            }
        }
    });
    if (handled) plKeyHoldRaf = requestAnimationFrame(plProcessKeyHold);
    else plKeyHoldRaf = null;
}

// ── Init ──────────────────────────────────────────────────────
function plInit() {
    plFrameImg = document.getElementById('pl-frame-img');
    plCanvas = document.getElementById('pl-canvas');
    plCtx = plCanvas.getContext('2d');

    // ResizeObserver to redraw canvas when container size changes
    const plVideoWrap = document.getElementById('pl-video-wrap');
    const plResizeObserver = new ResizeObserver(() => {
        if (activeTab === 'personlabel') {
            plDrawCanvas();
        }
    });
    plResizeObserver.observe(plVideoWrap);

    plCanvas.addEventListener('click', plHandleCanvasClick);
    plCanvas.addEventListener('mousedown', plHandleMouseDown);
    plCanvas.addEventListener('mousemove', plHandleMouseMove);
    plCanvas.addEventListener('mouseup', plHandleMouseUp);
    plCanvas.addEventListener('mouseleave', () => {
        if (plDrawingBox) {
            plDrawingBox = null;
            plDrawStart = null;
            plDrawCanvas();
        }
    });

    document.getElementById('pl-browse-btn').addEventListener('click', async () => {
        const res = await fetch('/api/browse_dir');
        const data = await res.json();
        if (data.ok) { document.getElementById('pl-workdir-input').value = data.path; plLoadClips(); }
    });
    document.getElementById('pl-load-btn').addEventListener('click', plLoadClips);
    document.getElementById('pl-detect-btn').addEventListener('click', plStartDetection);
    document.getElementById('pl-drawbox-btn').addEventListener('click', plToggleDrawBox);
    document.getElementById('pl-fillgaps-btn').addEventListener('click', plFillGaps);
    document.getElementById('pl-clear-btn').addEventListener('click', plClearClip);
    document.getElementById('pl-delete-btn').addEventListener('click', plDeleteClip);
    document.getElementById('pl-export-btn').addEventListener('click', plExport);
    document.getElementById('pl-save-btn').addEventListener('click', plSave);
    document.getElementById('pl-prev-btn').addEventListener('click', () => plNextHit(-1));
    document.getElementById('pl-next-btn').addEventListener('click', () => plNextHit(1));
    document.getElementById('pl-jump-p1').addEventListener('click', () => plJumpToMissing('p1'));
    document.getElementById('pl-jump-p2').addEventListener('click', () => plJumpToMissing('p2'));
    document.getElementById('pl-frame-input').addEventListener('change', e => {
        const v = parseInt(e.target.value); if (!isNaN(v)) plSetFrame(v);
    });

    plLoadClips();
}

// ── Keyboard ──────────────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (activeTab !== 'personlabel') return;
    if (e.target.tagName === 'INPUT') return;

    if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); plUndo(); return;
    }
    if (e.key === 'd' || e.key === 'D') {
        if (!e.metaKey && !e.ctrlKey) {
            e.preventDefault(); plToggleDrawBox(); return;
        }
    }
    if (e.key === ' ') {
        // Space: toggle HIT on current frame
        e.preventDefault();
        plToggleHit();
        return;
    }
    // 1 / 2 for jumping to missing P1/P2
    if (e.key === '1') {
        e.preventDefault();
        plJumpToMissing('p1');
        return;
    }
    if (e.key === '2') {
        e.preventDefault();
        plJumpToMissing('p2');
        return;
    }
    // Comma / Period for HIT frame navigation
    if (e.key === ',' || e.key === '<') {
        e.preventDefault();
        plNextHit(-1);
        return;
    }
    if (e.key === '.' || e.key === '>') {
        e.preventDefault();
        plNextHit(1);
        return;
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        if (e.repeat) return;
        // Frame stepping with hold-to-fast
        const state = plKeyHoldState[e.key];
        if (!state.active) {
            state.active = true;
            state.startTime = performance.now();
            state.lastRafTime = performance.now();
            plSetFrame(plCurrentFrame + (e.key === 'ArrowLeft' ? -1 : 1));
            if (!plKeyHoldRaf) plKeyHoldRaf = requestAnimationFrame(plProcessKeyHold);
        }
        return;
    } else if (e.key === 'ArrowUp') {
        e.preventDefault(); plSelectClip(plCurrentIdx - 1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault(); plSelectClip(plCurrentIdx + 1);
    }
});

document.addEventListener('keyup', e => {
    if (activeTab !== 'personlabel') return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        plKeyHoldState[e.key].active = false;
    }
});

plInit();

// ============================================================================
// HIT ANNOTATOR (hitlabeler) — hl prefix
// ============================================================================
let hlWorkdir = '/Users/harryzhang/git/tempopeak/datasets/v1/clips';
let hlClips = [];         // [{name, total_frames, hits}]
let hlCurrentIdx = -1;
let hlCurrentFrame = 0;
let hlTotalFrames = 0;
let hlHits = [];          // [frame, ...] sorted
let hlHistory = [];
let hlFrameImg = null;
let hlCanvas = null;
let hlCtx = null;

// ── Init ──────────────────────────────────────────────────────
function hlInit() {
    hlFrameImg = document.getElementById('hl-frame-img');
    hlCanvas = document.getElementById('hl-canvas');
    hlCtx = hlCanvas.getContext('2d');

    document.getElementById('hl-browse-btn').addEventListener('click', async () => {
        try {
            const res = await fetch('/api/browse_dir');
            if (res.ok) {
                const data = await res.json();
                if (data.ok && data.path) {
                    document.getElementById('hl-workdir-input').value = data.path;
                }
            }
        } catch (e) { console.error(e); }
    });

    document.getElementById('hl-load-btn').addEventListener('click', hlLoadClips);
    document.getElementById('hl-save-btn').addEventListener('click', hlSave);

    document.getElementById('hl-prev-btn').addEventListener('click', () => hlSetFrame(hlCurrentFrame - 1));
    document.getElementById('hl-next-btn').addEventListener('click', () => hlSetFrame(hlCurrentFrame + 1));

    document.getElementById('hl-frame-input').addEventListener('change', e => {
        const v = parseInt(e.target.value);
        if (!isNaN(v)) hlSetFrame(v);
    });

    hlLoadClips();
}

// ── Load Clips ──────────────────────────────────────────────────────
async function hlLoadClips() {
    hlWorkdir = document.getElementById('hl-workdir-input').value.trim();
    if (!hlWorkdir) return;

    const res = await fetch(`/api/hl/clips?workdir=${encodeURIComponent(hlWorkdir)}`);
    const data = await res.json();
    hlClips = data.clips || [];
    hlCurrentIdx = -1;
    hlRenderClipList();

    if (hlClips.length > 0) {
        hlSelectClip(0);
    } else {
        document.getElementById('hl-clip-name').textContent = 'No clips found';
        document.getElementById('hl-hit-count').textContent = '0 HITs';
        document.getElementById('hl-total-frames').textContent = '0';
    }
}

// ── Render Clip List ──────────────────────────────────────────────────────
function hlRenderClipList() {
    const el = document.getElementById('hl-clip-list');
    el.innerHTML = '';
    hlClips.forEach((clip, i) => {
        const item = document.createElement('div');
        item.className = 'clip-list-item' + (i === hlCurrentIdx ? ' active' : '') + (clip.hits && clip.hits.length > 0 ? ' annotated' : '');
        const hitCount = clip.hits ? clip.hits.length : 0;
        item.textContent = `${clip.name} ${hitCount > 0 ? `(${hitCount} HITs)` : '(—)'}`;
        item.onclick = () => hlSelectClip(i);
        el.appendChild(item);
    });
}

// ── Select Clip ──────────────────────────────────────────────────────
async function hlSelectClip(idx) {
    if (idx < 0 || idx >= hlClips.length) return;

    // Auto-save current clip before switching
    if (hlCurrentIdx >= 0 && hlCurrentIdx !== idx) {
        await hlSave();
    }

    hlCurrentIdx = idx;
    const clip = hlClips[idx];

    hlCurrentFrame = 0;
    hlHistory = [];
    hlHits = clip.hits ? [...clip.hits] : [];
    hlTotalFrames = clip.total_frames || 0;

    document.getElementById('hl-clip-name').textContent = clip.name;
    document.getElementById('hl-total-frames').textContent = hlTotalFrames;
    hlUpdateHitCount();

    hlSeekTo(0);
    hlRenderClipList();
}

// ── Seek ──────────────────────────────────────────────────────
function hlSeekTo(frame) {
    if (hlCurrentIdx < 0 || hlTotalFrames <= 0) return;
    hlCurrentFrame = Math.max(0, Math.min(frame, hlTotalFrames - 1));
    document.getElementById('hl-frame-input').value = hlCurrentFrame;
    const clip = hlClips[hlCurrentIdx];
    const url = `/api/pl/frame?workdir=${encodeURIComponent(hlWorkdir)}&name=${clip.name}&frame=${hlCurrentFrame}`;
    hlFrameImg.onload = () => hlDrawCanvas();
    hlFrameImg.src = url;

    // Prefetch nearby frames
    for (let delta of [-1, 1, 2, -2]) {
        const pf = hlCurrentFrame + delta;
        if (pf >= 0 && pf < hlTotalFrames) {
            const img = new Image();
            img.src = `/api/pl/frame?workdir=${encodeURIComponent(hlWorkdir)}&name=${clip.name}&frame=${pf}`;
        }
    }
}

function hlSetFrame(frame) {
    hlSeekTo(frame);
}

// ── Canvas Drawing ──────────────────────────────────────────────────────
function hlDrawCanvas() {
    if (!hlCanvas || !hlFrameImg || !hlFrameImg.naturalWidth) {
        return;
    }
    const wrap = document.getElementById('hl-frame-wrap');
    const canvasW = wrap.clientWidth, canvasH = wrap.clientHeight;
    const vw = hlFrameImg.naturalWidth, vh = hlFrameImg.naturalHeight;
    if (!vw || !vh || !canvasW || !canvasH) {
        if (activeTab === 'hitlabeler') {
            requestAnimationFrame(() => hlDrawCanvas());
        }
        return;
    }

    const scale = Math.min(canvasW / vw, canvasH / vh);
    const renderW = vw * scale, renderH = vh * scale;
    const offX = (canvasW - renderW) / 2;
    const offY = (canvasH - renderH) / 2;

    const dpr = window.devicePixelRatio || 1;
    hlCanvas.width = canvasW * dpr;
    hlCanvas.height = canvasH * dpr;
    hlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hlCtx.clearRect(0, 0, canvasW, canvasH);

    // Draw HIT marker if current frame is a HIT
    const isHit = hlHits.includes(hlCurrentFrame);
    if (isHit) {
        // Red triangle at top center
        hlCtx.fillStyle = '#ef4444';
        hlCtx.beginPath();
        hlCtx.moveTo(canvasW / 2, 6);
        hlCtx.lineTo(canvasW / 2 - 12, 30);
        hlCtx.lineTo(canvasW / 2 + 12, 30);
        hlCtx.closePath();
        hlCtx.fill();

        // "HIT" text
        hlCtx.font = 'bold 14px sans-serif';
        hlCtx.fillStyle = '#fff';
        hlCtx.textAlign = 'center';
        hlCtx.fillText('HIT', canvasW / 2, 26);
    }

    // Update hit indicator
    const el = document.getElementById('hl-hit-indicator');
    if (isHit) {
        el.textContent = 'HIT';
        el.style.color = '#ef4444';
    } else {
        el.textContent = '';
    }
}

// ── Toggle HIT ──────────────────────────────────────────────────────
function hlToggleHit() {
    if (hlCurrentIdx < 0) return;
    hlSaveHistory();

    const hitIdx = hlHits.indexOf(hlCurrentFrame);
    if (hitIdx >= 0) {
        hlHits.splice(hitIdx, 1);
    } else {
        // Insert sorted
        let insertIdx = 0;
        while (insertIdx < hlHits.length && hlHits[insertIdx] < hlCurrentFrame) {
            insertIdx++;
        }
        hlHits.splice(insertIdx, 0, hlCurrentFrame);
    }

    hlDrawCanvas();
    hlUpdateHitCount();
    hlRenderClipList();
}

// ── Delete HIT on current frame ──────────────────────────────────────
function hlDeleteHit() {
    if (hlCurrentIdx < 0) return;
    const hitIdx = hlHits.indexOf(hlCurrentFrame);
    if (hitIdx < 0) return; // No HIT to delete

    hlSaveHistory();
    hlHits.splice(hitIdx, 1);
    hlDrawCanvas();
    hlUpdateHitCount();
    hlRenderClipList();
}

function hlUpdateHitCount() {
    document.getElementById('hl-hit-count').textContent = `${hlHits.length} HITs`;
}

// ── History / Undo ──────────────────────────────────────────────────────
function hlSaveHistory() {
    hlHistory.push([...hlHits]);
    if (hlHistory.length > 50) hlHistory.shift();
}

function hlUndo() {
    if (hlHistory.length === 0) return;
    hlHits = hlHistory.pop();
    hlDrawCanvas();
    hlUpdateHitCount();
    hlRenderClipList();
}

// ── Save ──────────────────────────────────────────────────────
async function hlSave() {
    if (hlCurrentIdx < 0) return;
    const clip = hlClips[hlCurrentIdx];
    const btn = document.getElementById('hl-save-btn');
    btn.textContent = 'Saving...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/hl/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workdir: hlWorkdir,
                name: clip.name,
                hits: hlHits
            })
        });
        const data = await res.json();
        if (data.ok) {
            clip.hits = [...hlHits];
            hlRenderClipList();
            btn.textContent = 'Saved!';
            setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 2000);
        } else {
            throw new Error(data.detail || 'Save failed');
        }
    } catch (e) {
        alert('Save error: ' + e.message);
        btn.textContent = 'Save';
        btn.disabled = false;
    }
}

// ── Keyboard for HIT Annotator ──────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (activeTab !== 'hitlabeler') return;
    if (e.target.tagName === 'INPUT') return;

    if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        hlUndo();
        return;
    }
    if (e.key === ' ') {
        e.preventDefault();
        hlToggleHit();
        return;
    }
    if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        hlDeleteHit();
        return;
    }
    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        hlSetFrame(hlCurrentFrame - 1);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        hlSetFrame(hlCurrentFrame + 1);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        hlSelectClip(hlCurrentIdx - 1);
    } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        hlSelectClip(hlCurrentIdx + 1);
    }
});
// ── Warn before leaving with unsaved changes ──────────────────────────────────────────────────────
window.addEventListener("beforeunload", (e) => {
    if (plDirty) {
        e.preventDefault();
        e.returnValue = "You have unsaved changes. Are you sure you want to leave?";
        return e.returnValue;
    }
});


// ── Auto-save to memory every 10 seconds ──────────────────────────────────────────────────────
setInterval(() => {
    if (activeTab === "personlabel" && plCurrentIdx >= 0 && plDirty) {
        plSaveToMemory();
    }
}, 10000);

