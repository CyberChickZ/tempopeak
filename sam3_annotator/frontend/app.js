// State
let videoFile = null;
let jsonFile = null;
let npzFile = null;

let videoEl = document.getElementById('video-player');
let maskCanvas = document.getElementById('mask-canvas');
let uiCanvas = document.getElementById('ui-canvas');
let maskCtx = maskCanvas.getContext('2d');
let uiCtx = uiCanvas.getContext('2d');

let totalFrames = 0;
let fps = 30; // Configurable from UI
let currentFrame = 0;
let isPlaying = false;
let hasUnsavedChanges = false;

// Tracks data parsed strictly for timeline rendering:
// parsedTracks = { "123": { "7": { prompt: "ball", mask_idx: 10 }, ... } }
let parsedTracks = {};
// Track bounds for timeline presence bar:
// trackBounds = { "7": { color: "#ff0000", intervals: [[10, 50], [60, 100]] } }
let trackBounds = {};

// Caches
let maskImageCache = new Map(); // mask_idx -> HTMLImageElement
let frameInstancesCache = new Map(); // frame_idx -> UI response data

// Dynamic colors assigned to Object IDs
let idColors = {};

// Hardcoded fallback colors (like SAM Editor)
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

// ================= Scan and Load Setup =================
document.getElementById('browse-btn').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/browse_dir');
        if (res.ok) {
            const data = await res.json();
            if (data.ok && data.path) {
                document.getElementById('workdir-input').value = data.path;
                scanDirectory(); // Auto-scan
            }
        }
    } catch (e) { console.error("Browse Error:", e); }
});

window.addEventListener('DOMContentLoaded', scanDirectory);
document.getElementById('workdir-input').addEventListener('change', scanDirectory);

async function scanDirectory() {
    const workdir = document.getElementById('workdir-input').value.trim();
    if (!workdir) {
        return;
    }

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
        const listContainer = document.getElementById('video-list');
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
        }
    } catch (err) {
        alert(err.message);
        listContainer.innerHTML = '<div style="color: #64748b; font-size: 0.85rem; padding: 10px;">Scan failed.</div>';
    }
}

let selectedVideoName = null;
let activeVideoListItem = null;

function selectVideoItem(videoName, element) {
    if (selectedVideoName === videoName) return;

    if (hasUnsavedChanges) {
        if (!confirm("You have unsaved changes. Are you sure you want to load a different video and lose them?")) {
            return;
        }
    }

    if (activeVideoListItem) {
        activeVideoListItem.classList.remove('active');
    }
    element.classList.add('active');
    activeVideoListItem = element;
    selectedVideoName = videoName;
    hasUnsavedChanges = false;

    // Auto-trigger load
    loadVideoData(videoName);
}

async function loadVideoData(videoName) {

    // Stop playback if running
    if (isPlaying) {
        videoEl.pause();
        isPlaying = false;
        document.getElementById('play-pause-btn').innerText = "▶";
    }

    fps = parseInt(document.getElementById('fps-input').value) || 30;

    document.getElementById('loading-msg').innerText = "Loading data from backend...";

    try {
        // Load data in backend
        const res = await fetch('/api/load_by_name', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_name: videoName })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Backend load failed");
        }

        // Point to backend video stream
        videoEl.src = `/video?vid=${videoName}&t=${Date.now()}`;

        // Preload track data for timeline
        const dlRes = await fetch('/api/download_json');
        const jsonText = await dlRes.text();
        parsedTracks = JSON.parse(jsonText);
        buildTimelineData();

        videoEl.onloadedmetadata = () => {
            totalFrames = Math.max(...Object.keys(parsedTracks).map(Number), 0);

            // Calculate actual FPS natively from video duration and frame limits (e.g. 59.94 -> 60)
            if (videoEl.duration > 0 && totalFrames > 0) {
                const calculatedFps = Math.round(totalFrames / videoEl.duration);
                if (calculatedFps > 0) {
                    fps = calculatedFps;
                    document.getElementById('fps-input').value = fps;
                }
            }

            // Sync canvas resolution to internal physical pixels
            maskCanvas.width = videoEl.videoWidth;
            maskCanvas.height = videoEl.videoHeight;
            uiCanvas.width = videoEl.videoWidth;
            uiCanvas.height = videoEl.videoHeight;

            // Enforce perfect CSS layout fit for the wrapper based on video aspect ratio
            const wrapper = document.querySelector('.video-wrapper');
            wrapper.style.aspectRatio = `${videoEl.videoWidth} / ${videoEl.videoHeight}`;

            // (No JS ResizeObserver needed. CSS handles it natively!)

            renderTimelineTracks();

            // Initial render
            onFrameChange(0);
            document.getElementById('loading-msg').innerText = "Loaded successfully!";
            setTimeout(() => { document.getElementById('loading-msg').innerText = ""; }, 3000);
        };

        if (videoEl.readyState >= 1) {
            videoEl.onloadedmetadata(null);
        }

    } catch (err) {
        alert(err.message);
        document.getElementById('loading-msg').innerText = "Load failed.";
    }
}

// ================= Timeline & Data Processing =================
function buildTimelineData() {
    trackBounds = {};
    const frames = Object.keys(parsedTracks).map(Number).sort((a, b) => a - b);

    // Get all unique object IDs
    const objIds = new Set();
    frames.forEach(f => {
        Object.keys(parsedTracks[f]).forEach(oid => objIds.add(oid));
    });

    // Extremely simple presence (no merging gaps for MVP, just individual blocks or we can group contiguous)
    objIds.forEach(oid => {
        let presenceObjFrames = [];
        frames.forEach(f => {
            if (parsedTracks[f][oid]) {
                presenceObjFrames.push(f);
            }
        });

        let intervals = [];
        if (presenceObjFrames.length > 0) {
            let start = presenceObjFrames[0];
            let end = start;
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

        trackBounds[oid] = {
            intervals: intervals,
            defaultLabel: "unknown" // Will fetch from first occurrence
        };

        // Find first label
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
            const startFr = interval[0];
            const endFr = interval[1];

            const leftPct = (startFr / totalFrames) * 100;
            const widthPct = ((endFr - startFr + 1) / totalFrames) * 100;

            const bar = document.createElement('div');
            bar.className = 'track-presence';
            bar.style.left = `${leftPct}%`;
            bar.style.width = `${widthPct}%`;
            bar.style.backgroundColor = colorHex;
            row.appendChild(bar);
        });

        container.appendChild(row);
    });

    // Make timeline clickable
    container.addEventListener('click', (e) => {
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const clickedFrame = Math.round((x / rect.width) * totalFrames);
        setFrame(Math.min(Math.max(clickedFrame, 0), totalFrames));
    });
}

// ================= Playback & Navigation =================
function setFrame(frame) {
    if (frame < 0) frame = 0;
    if (frame > totalFrames) frame = totalFrames;

    // Add +0.005s to push the video clock reliably past the target frame's floating point boundary
    videoEl.currentTime = (frame / fps) + 0.005;

    onFrameChange(frame);
}

function playbackLoop() {
    if (!isPlaying) return;
    const frame = Math.round(videoEl.currentTime * fps);
    if (frame !== currentFrame && frame <= totalFrames) {
        onFrameChange(frame);
    }
    if (!videoEl.paused && !videoEl.ended) {
        requestAnimationFrame(playbackLoop);
    }
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

        // Hard sync back to exactly currentFrame on pause
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
        // Find current absolute time in seconds
        const currentTimeSec = videoEl.currentTime || 0;
        fps = val;

        // Re-evaluate what frame that time corresponds to now
        const newFrame = Math.round(currentTimeSec * fps);
        // Force the UI elements to update to the new frame mapping without affecting video time
        onFrameChange(newFrame);
    }
});

// Keyboard
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'a' || e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    if (e.key === 'd' || e.key === 'ArrowRight') setFrame(currentFrame + 1);
    if (e.key === ' ') {
        e.preventDefault();
        document.getElementById('play-pause-btn').click();
    }
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
    if (parsedTracks && parsedTracks[frame]) {
        instances = parsedTracks[frame];
    }

    renderSidebar(instances);
    await drawMasks(instances);
}

async function fetchMaskImage(mask_idx) {
    if (maskImageCache.has(mask_idx)) {
        return maskImageCache.get(mask_idx);
    }

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            maskImageCache.set(mask_idx, img);
            resolve(img);
        };
        img.onerror = reject;
        img.src = `/api/mask/${mask_idx}.png`;
    });
}

function tintCanvas(img, rgbArray, outCanvas, highlight = false) {
    outCanvas.width = img.width;
    outCanvas.height = img.height;
    const ctx = outCanvas.getContext('2d');

    // Draw white mask on transparent background
    ctx.drawImage(img, 0, 0);

    // Apply tint using hardware source-in
    ctx.globalCompositeOperation = 'source-in';
    const [r, g, b] = rgbArray;
    const alpha = highlight ? 0.7 : 0.4;
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.fillRect(0, 0, img.width, img.height);

    ctx.globalCompositeOperation = 'source-over';
    return outCanvas;
}

async function drawMasks(instances) {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height); // highlight layer

    // Hidden canvas for tinting calculation
    const tempCanvas = document.createElement('canvas');

    const promises = Object.entries(instances).map(async ([objId, info]) => {
        try {
            const img = await fetchMaskImage(info.mask_idx);
            return { objId, info, img };
        } catch (e) {
            console.error(`Failed to draw mask ${info.mask_idx}`, e);
            return null;
        }
    });

    const results = (await Promise.all(promises)).filter(r => r !== null);

    for (const { objId, info, img } of results) {
        const colorHex = getColorForId(objId);
        const color = hexToRgba(colorHex);
        const isHovered = hoveredObjectId === objId;

        const tinted = tintCanvas(img, color, tempCanvas, isHovered);
        maskCtx.drawImage(tinted, 0, 0);

        if (isHovered && info.box) {
            uiCtx.strokeStyle = `rgb(${color.join(',')})`;
            uiCtx.lineWidth = 2;
            uiCtx.strokeRect(info.box[0], info.box[1], info.box[2] - info.box[0], info.box[3] - info.box[1]);
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
                    <option value="ball" ${(info.label || info.prompt) === 'ball' ? 'selected' : ''}>Ball</option>
                    <option value="racket" ${(info.label || info.prompt) === 'racket' ? 'selected' : ''}>Racket</option>
                    <option value="unknown" ${(info.label || info.prompt) === 'unknown' || (!['ball', 'racket'].includes(info.label || info.prompt)) ? 'selected' : ''}>Unknown</option>
                </select>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Score: ${(info.tracker_score ?? info.score ?? 0).toFixed(3)}</div>
            </div>
        `;

        // Interactivity
        card.addEventListener('mouseenter', () => {
            hoveredObjectId = objId;
            card.classList.add('highlight');
            drawMasks(instances); // Redraw with highlight
        });
        card.addEventListener('mouseleave', () => {
            hoveredObjectId = null;
            card.classList.remove('highlight');
            drawMasks(instances);
        });

        const select = card.querySelector('.label-select');
        select.addEventListener('change', async (e) => {
            const newLabel = e.target.value;
            // Optimistic rendering
            if ('label' in info) info.label = newLabel; else info.prompt = newLabel;
            drawMasks(instances);

            // Backend update
            await fetch('/api/edit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame_idx: currentFrame, object_id: objId, prompt: newLabel })
            });
            // Update local timeline cache so re-renders of timeline reflect label changes
            const tracked = parsedTracks[currentFrame][objId];
            if ('label' in tracked) tracked.label = newLabel; else tracked.prompt = newLabel;
            const trackLabel = container.querySelector(`.track-label[data-oid="${objId}"]`);
            if (trackLabel) trackLabel.innerText = `Obj ${objId} (${newLabel})`;

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
            renderTimelineTracks(); // Refresh single gap in timeline
            onFrameChange(currentFrame); // Re-fetch and re-render current frame
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

            // Wipe from local frontend memory
            for (const frame in parsedTracks) {
                if (parsedTracks[frame][objId]) {
                    delete parsedTracks[frame][objId];
                }
            }

            // Full refresh
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
    // Trigger two downloads through backend
    window.open('/api/download_json', '_blank');

    // Adding slight delay for second download trigger to not get blocked
    setTimeout(() => {
        window.open('/api/download_npz', '_blank');
        hasUnsavedChanges = false; // Usually downloading counts as resolving dirty state
    }, 500);
});
