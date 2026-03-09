// State
window.onerror = function (msg, url, line, col, error) {
    alert("JS Error: " + msg + "\nLine: " + line);
    return false;
};

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
            // Auto-load first video
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
        // Auto-save before switching video
        try {
            await fetch('/api/save_overwrite', { method: 'POST' });
        } catch (e) {
            console.error("Auto save failed", e);
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
        delete parsedTracks["_meta"];  // Prevent NaN issues during loop processing
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

            // Initial render - Jump to peak hit score frame
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

        // Add Hit Score Curve
        let hasHitScore = false;
        let points = [];
        for (let f = 0; f <= totalFrames; f++) {
            if (parsedTracks[f] && parsedTracks[f][oid]) {
                const hs = parsedTracks[f][oid].hit_score;
                if (hs !== undefined) {
                    hasHitScore = true;
                    points.push({ f: f, s: hs });
                }
            }
        }

        if (hasHitScore && points.length > 0) {
            const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.style.position = "absolute";
            svg.style.left = "0";
            svg.style.top = "0";
            svg.style.width = "100%";
            svg.style.height = "100%";
            svg.style.pointerEvents = "none";
            svg.setAttribute("preserveAspectRatio", "none");
            svg.setAttribute("viewBox", "0 0 1000 100");

            let d = "";
            let started = false;

            for (let p of points) {
                const x = (p.f / totalFrames) * 1000;
                const y = 100 - (p.s * 100);
                if (!started) {
                    d += `M ${x} ${y} `;
                    started = true;
                } else {
                    d += `L ${x} ${y} `;
                }
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
        } else {
            console.error("Failed to edit hit score");
        }
    } catch (err) {
        console.error("Error applying hit score:", err);
    }
}

document.getElementById('calibrate-btn').addEventListener('click', applyHitScoreCalibration);

// Keyboard
document.addEventListener('keydown', async e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'a' || e.key === 'ArrowLeft') setFrame(currentFrame - 1);
    if (e.key === 'd' || e.key === 'ArrowRight') setFrame(currentFrame + 1);
    if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (activeVideoListItem && activeVideoListItem.previousElementSibling) {
            activeVideoListItem.previousElementSibling.click();
        }
    }
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (activeVideoListItem && activeVideoListItem.nextElementSibling) {
            activeVideoListItem.nextElementSibling.click();
        }
    }
    if (e.key === ' ') {
        e.preventDefault();
        await applyHitScoreCalibration();
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

        if (info.box) {
            uiCtx.strokeStyle = `rgb(${color.join(',')})`;
            uiCtx.globalAlpha = isHovered ? 1.0 : 0.4;
            uiCtx.lineWidth = isHovered ? 2 : 1;
            uiCtx.strokeRect(info.box[0], info.box[1], info.box[2] - info.box[0], info.box[3] - info.box[1]);
            uiCtx.globalAlpha = 1.0; // reset
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

            // Backend update (update entire track)
            await fetch('/api/edit_track', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ object_id: objId, prompt: newLabel })
            });

            // Update local timeline cache so re-renders of timeline reflect label changes globally
            for (const f in parsedTracks) {
                if (parsedTracks[f][objId]) {
                    if ('label' in parsedTracks[f][objId]) {
                        parsedTracks[f][objId].label = newLabel;
                    } else {
                        parsedTracks[f][objId].prompt = newLabel;
                    }
                }
            }
            if (trackBounds[objId]) trackBounds[objId].defaultLabel = newLabel;

            const listContainer = document.getElementById('tracks-container');
            if (listContainer) {
                const trackLabel = listContainer.querySelector(`.track-label[data-oid="${objId}"]`);
                if (trackLabel) trackLabel.innerText = `Obj ${objId} (${newLabel})`;
            }

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

// ============================================================================
// CLIP REVIEW TAB LOGIC
// ============================================================================

const tabBtns = document.querySelectorAll('.tab-btn');
const screens = document.querySelectorAll('.screen');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        screens.forEach(s => s.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(`${tabId}-screen`).classList.add('active');
        
        // Pause annotator video if switching away
        if (tabId === 'clipreview') {
            if (elements.video && !elements.video.paused) elements.video.pause();
            loadClipsList(); // auto refresh list
        } else {
            if (clipVideoEl && !clipVideoEl.paused) clipVideoEl.pause();
        }
    });
});

let clips = [];
let currentClipIdx = -1;
let clipFrame = 0;
let clipTotalFrames = 0;
let clipEventSource = null;

const clipListEl = document.getElementById('clip-list');
const clipProgressEl = document.getElementById('clip-progress');
const clipCanvas = document.getElementById('clip-canvas');
const clipCtx = clipCanvas.getContext('2d');
const clipVideoEl = document.getElementById('clip-video-el');
const clipInfoEl = document.getElementById('clip-info');
const clipFilmstripEl = document.getElementById('clip-filmstrip');

document.getElementById('clip-start-btn').addEventListener('click', async () => {
    const folder = document.getElementById('clip-folder-input').value.trim();
    if (!folder) return alert("Please enter a folder path");

    try {
        const res = await fetch('/api/clips/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder })
        });
        if (!res.ok) throw new Error(await res.text());

        clipProgressEl.textContent = "Starting...";
        if (clipEventSource) clipEventSource.close();
        clipEventSource = new EventSource('/api/clips/stream');
        
        clipEventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.type === "ping") return;
            if (data.type === "progress") {
                clipProgressEl.textContent = `[YOLO] ${data.video_name}: frame ${data.frame}/${data.total}`;
            } else if (data.type === "clip") {
                clipProgressEl.textContent = `Cut clip ${data.clip_id}`;
                loadClipsList();
            } else if (data.type === "skip") {
                clipProgressEl.textContent = `Skipped ${data.video_name}: ${data.reason}`;
            } else if (data.type === "done") {
                clipProgressEl.textContent = "Extraction Complete";
                clipEventSource.close();
                clipEventSource = null;
                loadClipsList();
            } else if (data.type === "error") {
                clipProgressEl.textContent = `Error: ${data.message}`;
                clipEventSource.close();
                clipEventSource = null;
            }
        };
    } catch (e) {
        alert(e.message);
    }
});

async function loadClipsList() {
    try {
        const res = await fetch('/api/clips/list');
        const data = await res.json();
        clips = data.clips;
        renderClipList();
        if (clips.length > 0 && currentClipIdx === -1) {
            selectClip(0);
        } else if (clips.length === 0) {
            clipListEl.innerHTML = '<div style="padding:10px;color:#94a3b8;font-size:0.85rem">No clips</div>';
            clearClipView();
        }
    } catch(e) { console.error("Error loading clips:", e); }
}

function renderClipList() {
    clipListEl.innerHTML = '';
    clips.forEach((c, idx) => {
        const div = document.createElement('div');
        div.className = `clip-list-item ${idx === currentClipIdx ? 'selected' : ''} ${c.reviewed ? 'annotated' : ''}`;
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = c.id;
        
        const badgeSpan = document.createElement('span');
        badgeSpan.className = 'hit-badge';
        if (c.reviewed) {
            badgeSpan.textContent = c.label === 'hit' ? `✓ Hit f${c.hit_frame}` : (c.label === 'reject' ? '❌' : '');
        }

        div.appendChild(nameSpan);
        div.appendChild(badgeSpan);
        div.onclick = () => selectClip(idx);
        clipListEl.appendChild(div);
    });
}

function clearClipView() {
    clipCtx.clearRect(0, 0, clipCanvas.width, clipCanvas.height);
    clipInfoEl.textContent = "No clip loaded";
    clipFilmstripEl.innerHTML = "";
    clipVideoEl.removeAttribute('src');
    clipVideoEl.load();
    currentClipIdx = -1;
}

async function selectClip(idx) {
    if (idx < 0 || idx >= clips.length) return;
    currentClipIdx = idx;
    const clip = clips[idx];
    
    renderClipList();
    
    clipInfoEl.textContent = `Loading ${clip.id}...`;
    clipVideoEl.src = `/api/clips/video/${clip.id}`;
    
    clipVideoEl.onloadedmetadata = () => {
        clipCanvas.width = clipVideoEl.videoWidth;
        clipCanvas.height = clipVideoEl.videoHeight;
        clipTotalFrames = clip.num_frames;
        clipFrame = clip.reviewed && clip.label === 'hit' ? clip.hit_frame : Math.floor(clipTotalFrames / 2);
        
        clipInfoEl.textContent = `${clip.id} • ${clipTotalFrames} frames • Status: ${clip.reviewed ? clip.label + (clip.hit_frame !== null ? ' (f'+clip.hit_frame+')' : '') : 'unreviewed'}`;
        
        renderFilmstrip();
        seekClipTo(clipFrame);
    };
}

function seekClipTo(f) {
    if (f < 0) f = 0;
    if (f >= clipTotalFrames) f = clipTotalFrames - 1;
    clipFrame = f;
    
    // Convert frame to time (assuming source fps logic handled by backend, we just estimate time)
    // Actually HTMLVideoElement frame seeking is unreliable without knowing exact fps.
    // For local mp4 without audio, fps is usually 30.
    // Better: use requestAnimationFrame to draw whenever seeked.
    const fps = 30; // Approximation for UI seeking
    clipVideoEl.currentTime = clipFrame / fps;
    
    updateFilmstripHighlight();
}

clipVideoEl.addEventListener('seeked', () => {
    clipCtx.drawImage(clipVideoEl, 0, 0, clipCanvas.width, clipCanvas.height);
    
    // Draw crosshair/info
    clipCtx.fillStyle = 'white';
    clipCtx.font = '24px sans-serif';
    clipCtx.fillText(`Frame: ${clipFrame}`, 20, 40);
});

function renderFilmstrip() {
    clipFilmstripEl.innerHTML = '';
    // Draw 1 thumbnail for every frame is too much if >30. Just draw small divs
    for (let i = 0; i < clipTotalFrames; i++) {
        const c = document.createElement('canvas');
        c.width = 80; c.height = 55;
        c.onclick = () => seekClipTo(i);
        // We defer drawing the actual frame content to avoid 30 seeks. Just leave it blank/grey for speed, 
        // or just let it act as a timeline track.
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#334155';
        ctx.fillRect(0,0,80,55);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans';
        ctx.fillText(i.toString(), 30, 32);
        
        clipFilmstripEl.appendChild(c);
    }
}

function updateFilmstripHighlight() {
    const canvases = clipFilmstripEl.querySelectorAll('canvas');
    const clip = clips[currentClipIdx];
    canvases.forEach((c, i) => {
        c.className = '';
        if (i === clipFrame) c.classList.add('active-frame');
        if (clip && clip.reviewed && clip.label === 'hit' && clip.hit_frame === i) {
            c.classList.add('hit-frame');
        }
    });
}

async function markClipHit() {
    if (currentClipIdx < 0) return;
    const clip = clips[currentClipIdx];
    try {
        await fetch('/api/clips/annotate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({clip_id: clip.id, hit_frame: clipFrame})
        });
        clip.reviewed = true;
        clip.label = 'hit';
        clip.hit_frame = clipFrame;
        renderClipList();
        updateFilmstripHighlight();
        clipInfoEl.textContent = `${clip.id} • ${clipTotalFrames} frames • Status: hit (f${clip.hit_frame})`;
        
        // Auto advance after short delay
        setTimeout(() => {
            if (currentClipIdx < clips.length - 1 && document.getElementById('clipreview-screen').classList.contains('active')) {
                selectClip(currentClipIdx + 1);
            }
        }, 300);
    } catch(e) { console.error(e); }
}

async function markClipReject() {
    if (currentClipIdx < 0) return;
    const clip = clips[currentClipIdx];
    try {
        await fetch('/api/clips/reject', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({clip_id: clip.id})
        });
        clip.reviewed = true;
        clip.label = 'reject';
        clip.hit_frame = null;
        renderClipList();
        clipInfoEl.textContent = `${clip.id} • ${clipTotalFrames} frames • Status: reject`;
        
        setTimeout(() => {
            if (currentClipIdx < clips.length - 1 && document.getElementById('clipreview-screen').classList.contains('active')) {
                selectClip(currentClipIdx + 1);
            }
        }, 150);
    } catch(e) { console.error(e); }
}

async function markClipUndo() {
    if (currentClipIdx < 0) return;
    const clip = clips[currentClipIdx];
    try {
        await fetch('/api/clips/undo', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({clip_id: clip.id})
        });
        clip.reviewed = false;
        clip.label = null;
        clip.hit_frame = null;
        renderClipList();
        updateFilmstripHighlight();
        clipInfoEl.textContent = `${clip.id} • ${clipTotalFrames} frames • Status: unreviewed`;
    } catch(e) { console.error(e); }
}

document.getElementById('clip-export-btn').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/clips/export', {method: 'POST'});
        const data = await res.json();
        if (data.ok) {
            alert(`Exported successfully!\nTotal: ${data.total_exported}\nTo: ${data.out_dir}`);
        } else {
            alert(data.detail || 'Export failed');
        }
    } catch(e) { alert(e.message); }
});

// Clip Review Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    const isClipTab = document.getElementById('clipreview-screen').classList.contains('active');
    if (!isClipTab) return;
    if (e.target.tagName === 'INPUT') return;
    
    if (e.key === 'ArrowRight') {
        seekClipTo(clipFrame + 1);
        e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
        seekClipTo(clipFrame - 1);
        e.preventDefault();
    } else if (e.key === 'ArrowDown') {
        selectClip(currentClipIdx + 1);
        e.preventDefault();
    } else if (e.key === 'ArrowUp') {
        selectClip(currentClipIdx - 1);
        e.preventDefault();
    } else if (e.key === ' ') {
        markClipHit();
        e.preventDefault();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
        // Backspace to undo, Delete to reject (macOS fn+Backspace = Delete)
        if (e.key === 'Backspace') markClipUndo();
        else markClipReject();
        e.preventDefault();
    } else if (e.key === 'r') { // fallback for reject if no del key
        markClipReject();
    }
});
