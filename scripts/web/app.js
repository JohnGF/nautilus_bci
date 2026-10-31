/**
 * Nautilus BCI Studio - Unified Web Interface Engine
 * Complete client-side architecture for 2D Scalp Headmap, 60 FPS Oscilloscope,
 * Three.js 6-DoF Spatial Neurofeedback, and BIDS Management.
 */

// ── 1. VENDOR 10-20 MONTAGE (GPYPE_32) COORDINATES ──
// Normalized to radius 1.0 (Nose at Top: Y > 0, Ears at X = ±1.0)
const GPYPE_32_LAYOUT = [
    { name: "Fp1",  x: -0.30, y:  0.80, region: "Left Frontopolar" },
    { name: "Fp2",  x:  0.30, y:  0.80, region: "Right Frontopolar" },
    { name: "AF3",  x: -0.25, y:  0.66, region: "Left Anterior Frontal" },
    { name: "AF4",  x:  0.25, y:  0.66, region: "Right Anterior Frontal" },
    { name: "F7",   x: -0.75, y:  0.58, region: "Left Inferior Frontal" },
    { name: "F3",   x: -0.38, y:  0.50, region: "Left Mid-Frontal" },
    { name: "Fz",   x:  0.00, y:  0.52, region: "Midline Frontal" },
    { name: "F4",   x:  0.38, y:  0.50, region: "Right Mid-Frontal" },
    { name: "F8",   x:  0.75, y:  0.58, region: "Right Inferior Frontal" },
    { name: "FC5",  x: -0.65, y:  0.28, region: "Left Frontocentral" },
    { name: "FC1",  x: -0.25, y:  0.26, region: "Left Pre-Motor" },
    { name: "FC2",  x:  0.25, y:  0.26, region: "Right Pre-Motor" },
    { name: "FC6",  x:  0.65, y:  0.28, region: "Right Frontocentral" },
    { name: "T7",   x: -0.85, y:  0.00, region: "Left Mid-Temporal" },
    { name: "C3",   x: -0.45, y:  0.00, region: "Left Motor Cortex (Right Hand)" },
    { name: "Cz",   x:  0.00, y:  0.00, region: "Midline Motor Vertex (Feet)" },
    { name: "C4",   x:  0.45, y:  0.00, region: "Right Motor Cortex (Left Hand)" },
    { name: "T8",   x:  0.85, y:  0.00, region: "Right Mid-Temporal" },
    { name: "CP5",  x: -0.65, y: -0.28, region: "Left Centroparietal" },
    { name: "CP1",  x: -0.25, y: -0.26, region: "Left Somatosensory" },
    { name: "CP2",  x:  0.25, y: -0.26, region: "Right Somatosensory" },
    { name: "CP6",  x:  0.65, y: -0.28, region: "Right Centroparietal" },
    { name: "P7",   x: -0.75, y: -0.58, region: "Left Inferior Parietal" },
    { name: "P3",   x: -0.38, y: -0.50, region: "Left Mid-Parietal" },
    { name: "Pz",   x:  0.00, y: -0.52, region: "Midline Parietal" },
    { name: "P4",   x:  0.38, y: -0.50, region: "Right Mid-Parietal" },
    { name: "P8",   x:  0.75, y: -0.58, region: "Right Inferior Parietal" },
    { name: "PO7",  x: -0.60, y: -0.72, region: "Left Parieto-Occipital" },
    { name: "PO3",  x: -0.25, y: -0.70, region: "Left Parieto-Occipital" },
    { name: "PO4",  x:  0.25, y: -0.70, region: "Right Parieto-Occipital" },
    { name: "PO8",  x:  0.60, y: -0.72, region: "Right Parieto-Occipital" },
    { name: "Oz",   x:  0.00, y: -0.80, region: "Midline Occipital (Visual)" }
];

// ── 2. GLOBAL STATE & TELEMETRY MODEL ──
const state = {
    connected: false,
    isSimulated: true,
    battery: 80.0,
    channels: {},
    retouchPins: [],
    flatPins: [],
    medianRms: 22.4,
    activeTab: 'mirror',
    
    // Waveform Scope Buffers (2 seconds @ 60 FPS)
    scopeSamples: 180,
    scopeData: {
        Cz: new Float32Array(180),
        C3: new Float32Array(180),
        C4: new Float32Array(180),
        Oz: new Float32Array(180)
    },
    
    // Brain Rhythms
    powerAlpha: 45,
    powerBeta: 30,
    powerDelta: 25,
    
    // Neurofeedback
    mentalDrive: 50,
    lateralization: 0.0,
    bciState: "Standby",
    
    // BIDS Recording
    recording: false,
    recordStartTime: 0,
    recordSamples: 0,
    recordEvents: 0
};

// Initialize channel state table
GPYPE_32_LAYOUT.forEach(ch => {
    state.channels[ch.name] = {
        name: ch.name,
        region: ch.region,
        status: 'clean', // 'clean', 'noisy', 'flat', 'railed'
        rms: 20.0,
        ptp: 65.0
    };
});

// ── 3. SVG 2D SCALP HEADMAP RENDERER ──
function initSvgHeadmaps() {
    const containers = [
        document.getElementById('mirrorHeadmapSvg'),
        document.getElementById('workstationHeadmapSvg')
    ];

    containers.forEach(svg => {
        if (!svg) return;
        svg.innerHTML = '';
        const size = 480;
        const center = size / 2;
        const radius = size * 0.42;

        svg.setAttribute('viewBox', `0 0 ${size} ${size}`);

        // 1. Draw Head Circle
        const headCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        headCircle.setAttribute('cx', center);
        headCircle.setAttribute('cy', center);
        headCircle.setAttribute('r', radius);
        headCircle.setAttribute('fill', '#101622');
        headCircle.setAttribute('stroke', '#26334D');
        headCircle.setAttribute('stroke-width', '3');
        svg.appendChild(headCircle);

        // 2. Draw Nose (Pointing North)
        const nose = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        const noseTop = center - radius - 24;
        nose.setAttribute('points', `${center-20},${center-radius+2} ${center},${noseTop} ${center+20},${center-radius+2}`);
        nose.setAttribute('fill', '#161D2B');
        nose.setAttribute('stroke', '#26334D');
        nose.setAttribute('stroke-width', '3');
        svg.appendChild(nose);

        // 3. Draw Ears
        const earL = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        earL.setAttribute('d', `M ${center-radius-2} ${center-26} C ${center-radius-22} ${center-15}, ${center-radius-22} ${center+15}, ${center-radius-2} ${center+26}`);
        earL.setAttribute('fill', 'none');
        earL.setAttribute('stroke', '#26334D');
        earL.setAttribute('stroke-width', '3');
        svg.appendChild(earL);

        const earR = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        earR.setAttribute('d', `M ${center+radius+2} ${center-26} C ${center+radius+22} ${center-15}, ${center+radius+22} ${center+15}, ${center+radius+2} ${center+26}`);
        earR.setAttribute('fill', 'none');
        earR.setAttribute('stroke', '#26334D');
        earR.setAttribute('stroke-width', '3');
        svg.appendChild(earR);

        // 4. Directional Watermarks
        const labelNose = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelNose.setAttribute('x', center);
        labelNose.setAttribute('y', center - radius + 22);
        labelNose.setAttribute('text-anchor', 'middle');
        labelNose.setAttribute('fill', '#4A5568');
        labelNose.setAttribute('font-size', '11');
        labelNose.setAttribute('font-weight', 'bold');
        labelNose.textContent = 'FRONT (NOSE)';
        svg.appendChild(labelNose);

        // 5. Draw 32 Electrode Pins
        const dotRadius = 15;
        GPYPE_32_LAYOUT.forEach(ch => {
            const px = center + ch.x * radius * 0.96;
            const py = center - ch.y * radius * 0.96; // Invert Y for screen

            const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            group.setAttribute('class', `electrode-group electrode-${ch.name}`);
            group.setAttribute('data-channel', ch.name);

            const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot.setAttribute('cx', px);
            dot.setAttribute('cy', py);
            dot.setAttribute('r', dotRadius);
            dot.setAttribute('fill', '#3FB950'); // Clean green default
            dot.setAttribute('stroke', '#FFFFFF');
            dot.setAttribute('stroke-width', '1.5');
            dot.setAttribute('class', `electrode-dot dot-${ch.name}`);
            group.appendChild(dot);

            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', px);
            text.setAttribute('y', py + 4);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', '#FFFFFF');
            text.setAttribute('font-size', '9');
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('pointer-events', 'none');
            text.textContent = ch.name;
            group.appendChild(text);

            group.addEventListener('click', () => showChannelDetail(ch.name));
            svg.appendChild(group);
        });
    });
}

function updateHeadmapVisuals() {
    const statusColors = {
        clean:  '#3FB950', // Green
        noisy:  '#D29922', // Yellow / Retouch
        flat:   '#FF7B72', // Red
        railed: '#FFA657'  // Orange
    };

    GPYPE_32_LAYOUT.forEach(ch => {
        const info = state.channels[ch.name] || { status: 'clean' };
        const color = statusColors[info.status] || '#3FB950';

        document.querySelectorAll(`.dot-${ch.name}`).forEach(dot => {
            dot.setAttribute('fill', color);
            if (info.status === 'noisy') {
                dot.classList.add('noisy');
            } else {
                dot.classList.remove('noisy');
            }
        });
    });

    // Update Retouch Alert Banners
    const mirrorBanner = document.getElementById('mirrorRetouchBanner');
    const mirrorTitle = document.getElementById('mirrorAlertTitle');
    const mirrorList = document.getElementById('mirrorAlertPins');
    const mirrorHint = document.getElementById('mirrorAlertHint');

    if (state.retouchPins.length > 0 || state.flatPins.length > 0) {
        mirrorBanner.classList.remove('all-clean');
        const warnItems = [];
        if (state.retouchPins.length > 0) {
            warnItems.push(`RETOUCH (${state.retouchPins.length} pins): ${state.retouchPins.join(', ')}`);
        }
        if (state.flatPins.length > 0) {
            warnItems.push(`FLATLINE (${state.flatPins.length} pins): ${state.flatPins.join(', ')}`);
        }
        mirrorTitle.textContent = `⚠️ ADJUST SENSORS IN MIRROR`;
        mirrorList.textContent = warnItems.join(' | ');
        mirrorHint.textContent = `Part hair gently and twist gold pins so they touch the scalp directly.`;
    } else {
        mirrorBanner.classList.add('all-clean');
        mirrorTitle.textContent = `✅ ALL 32 ELECTRODES OPTIMAL`;
        mirrorList.textContent = `Median Noise Floor: ${state.medianRms.toFixed(1)} µV | Sensorimotor C3, Cz, C4 Clean`;
        mirrorHint.textContent = `Excellent contact through hair. Ready for BIDS Recording and 3D Task!`;
    }
}

function showChannelDetail(name) {
    const ch = state.channels[name];
    if (!ch) return;
    showToast(`📍 ${ch.name} (${ch.region}): RMS ${ch.rms.toFixed(1)}µV [${ch.status.toUpperCase()}]`);
}

// ── 4. 60 FPS HTML5 CANVAS LIVE OSCILLOSCOPE ──
let scopeCanvas, scopeCtx;
function initScopeCanvas() {
    scopeCanvas = document.getElementById('scopeCanvas');
    if (!scopeCanvas) return;
    scopeCtx = scopeCanvas.getContext('2d');
    resizeScopeCanvas();
    window.addEventListener('resize', resizeScopeCanvas);
}

function resizeScopeCanvas() {
    if (!scopeCanvas) return;
    const rect = scopeCanvas.getBoundingClientRect();
    scopeCanvas.width = rect.width * window.devicePixelRatio;
    scopeCanvas.height = rect.height * window.devicePixelRatio;
}

function renderScope() {
    if (!scopeCtx || !scopeCanvas) return;
    const w = scopeCanvas.width;
    const h = scopeCanvas.height;

    // Clear background
    scopeCtx.fillStyle = '#05080E';
    scopeCtx.fillRect(0, 0, w, h);

    // Draw Grid Lines
    scopeCtx.strokeStyle = 'rgba(38, 51, 77, 0.4)';
    scopeCtx.lineWidth = 1;
    for (let x = 0; x < w; x += w / 8) {
        scopeCtx.beginPath();
        scopeCtx.moveTo(x, 0);
        scopeCtx.lineTo(x, h);
        scopeCtx.stroke();
    }

    const channelKeys = [
        { name: 'Cz', color: '#3FB950', baseline: h * 0.20 },
        { name: 'C3', color: '#38EF7D', baseline: h * 0.40 },
        { name: 'C4', color: '#BC85FF', baseline: h * 0.60 },
        { name: 'Oz', color: '#D29922', baseline: h * 0.80 }
    ];

    const nSamples = state.scopeSamples;
    const dx = w / (nSamples - 1);
    const scale = 1.2; // Gain multiplier

    channelKeys.forEach(cfg => {
        const buf = state.scopeData[cfg.name];
        if (!buf) return;

        // Baseline reference line
        scopeCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        scopeCtx.beginPath();
        scopeCtx.moveTo(0, cfg.baseline);
        scopeCtx.lineTo(w, cfg.baseline);
        scopeCtx.stroke();

        // Label
        scopeCtx.fillStyle = cfg.color;
        scopeCtx.font = `${11 * window.devicePixelRatio}px -apple-system, sans-serif`;
        scopeCtx.fillText(cfg.name, 10, cfg.baseline - 6);

        // Trace line
        scopeCtx.strokeStyle = cfg.color;
        scopeCtx.lineWidth = 1.8 * window.devicePixelRatio;
        scopeCtx.beginPath();
        for (let i = 0; i < nSamples; i++) {
            const x = i * dx;
            const y = cfg.baseline - (buf[i] * scale);
            if (i === 0) scopeCtx.moveTo(x, y);
            else scopeCtx.lineTo(x, y);
        }
        scopeCtx.stroke();
    });

    requestAnimationFrame(renderScope);
}

// ── 5. THREE.JS 6-DOF SPATIAL NEUROFEEDBACK SCENE ──
let threeScene, threeCamera, threeRenderer, vehicleMesh;
function initThreeJsTask() {
    const container = document.getElementById('threeViewportWrap');
    const canvas = document.getElementById('taskGlCanvas');
    if (!canvas || !container) return;

    threeScene = new THREE.Scene();
    threeScene.fog = new THREE.FogExp2(0x030508, 0.0015);

    threeCamera = new THREE.PerspectiveCamera(65, container.clientWidth / container.clientHeight, 0.1, 1000);
    threeCamera.position.set(0, 5, 14);

    threeRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    threeRenderer.setSize(container.clientWidth, container.clientHeight);
    threeRenderer.setPixelRatio(window.devicePixelRatio);

    // Starfield
    const starGeo = new THREE.BufferGeometry();
    const starCount = 1200;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
        starPos[i] = (Math.random() - 0.5) * 400;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({ color: 0x58A6FF, size: 1.2, transparent: true, opacity: 0.7 });
    const starField = new THREE.Points(starGeo, starMat);
    threeScene.add(starField);

    // Lighting
    const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
    threeScene.add(ambLight);
    const dirLight = new THREE.DirectionalLight(0x58A6FF, 1.2);
    dirLight.position.set(10, 20, 15);
    threeScene.add(dirLight);

    // Submersible Nautilus Vehicle Mesh
    const hullGeo = new THREE.ConeGeometry(1.6, 5.0, 16);
    hullGeo.rotateX(Math.PI / 2);
    const hullMat = new THREE.MeshStandardMaterial({ color: 0x1F6FEB, metalness: 0.8, roughness: 0.2 });
    vehicleMesh = new THREE.Mesh(hullGeo, hullMat);

    // Fin details
    const finGeo = new THREE.BoxGeometry(4.2, 0.12, 1.2);
    const finMat = new THREE.MeshStandardMaterial({ color: 0x38EF7D, metalness: 0.6, roughness: 0.3 });
    const wings = new THREE.Mesh(finGeo, finMat);
    wings.position.set(0, 0, -0.6);
    vehicleMesh.add(wings);

    threeScene.add(vehicleMesh);
    threeCamera.lookAt(vehicleMesh.position);

    window.addEventListener('resize', () => {
        if (!container) return;
        threeCamera.aspect = container.clientWidth / container.clientHeight;
        threeCamera.updateProjectionMatrix();
        threeRenderer.setSize(container.clientWidth, container.clientHeight);
    });

    animateThreeJs();
}

let vehicleVelocity = { x: 0, y: 0, z: 0 };
let vehicleRotation = { pitch: 0, yaw: 0, roll: 0 };
function animateThreeJs() {
    requestAnimationFrame(animateThreeJs);
    if (!threeRenderer || !threeScene || !vehicleMesh) return;

    // Apply mental drive forward surge
    const surge = (state.mentalDrive / 100.0) * 0.15;
    vehicleMesh.position.z -= surge;

    // Apply lateral SMR banking
    vehicleMesh.position.x += state.lateralization * 0.12;
    vehicleMesh.rotation.z = -state.lateralization * 0.55;

    // Camera follow
    threeCamera.position.x = vehicleMesh.position.x * 0.6;
    threeCamera.position.z = vehicleMesh.position.z + 14;

    threeRenderer.render(threeScene, threeCamera);
}

// ── 6. REALISTIC STREAMING & MOCK TELEMETRY ENGINE ──
let simPhase = 0;
function runSimulatedTelemetry() {
    simPhase += 0.05;

    // Generate waveforms (Alpha bursts + SMR)
    const dt = 1.0 / 60.0;
    const t = simPhase;

    for (let i = 0; i < state.scopeSamples - 1; i++) {
        state.scopeData.Cz[i] = state.scopeData.Cz[i + 1];
        state.scopeData.C3[i] = state.scopeData.C3[i + 1];
        state.scopeData.C4[i] = state.scopeData.C4[i + 1];
        state.scopeData.Oz[i] = state.scopeData.Oz[i + 1];
    }

    const last = state.scopeSamples - 1;
    state.scopeData.Cz[last] = 18 * Math.sin(2 * Math.PI * 10 * t) + (Math.random() - 0.5) * 6;
    state.scopeData.C3[last] = 22 * Math.sin(2 * Math.PI * 10.5 * t) + (Math.random() - 0.5) * 7;
    state.scopeData.C4[last] = 20 * Math.sin(2 * Math.PI * 9.8 * t) + (Math.random() - 0.5) * 7;
    state.scopeData.Oz[last] = 30 * Math.sin(2 * Math.PI * 10.0 * t) + (Math.random() - 0.5) * 5;

    // Update Band powers
    state.powerAlpha = Math.min(100, Math.max(10, Math.round(50 + 25 * Math.sin(t * 0.8))));
    state.powerBeta  = Math.min(100, Math.max(10, Math.round(30 + 15 * Math.cos(t * 1.2))));
    state.powerDelta = Math.min(100, Math.max(5,  Math.round(20 + 8 * Math.sin(t * 0.4))));

    document.getElementById('barFillAlpha').style.width = `${state.powerAlpha}%`;
    document.getElementById('barValAlpha').textContent = `${state.powerAlpha}%`;
    document.getElementById('barFillBeta').style.width = `${state.powerBeta}%`;
    document.getElementById('barValBeta').textContent = `${state.powerBeta}%`;
    document.getElementById('barFillDelta').style.width = `${state.powerDelta}%`;
    document.getElementById('barValDelta').textContent = `${state.powerDelta}%`;

    // SMR Feedback Mental Drive
    state.mentalDrive = Math.min(100, Math.max(0, Math.round(state.powerBeta * 1.8)));
    state.lateralization = (state.powerAlpha > 60) ? 0.35 : -0.20;
    
    document.getElementById('hudDriveFill').style.width = `${state.mentalDrive}%`;
    document.getElementById('hudDriveText').textContent = `Drive: ${state.mentalDrive}%`;

    // Update Channel Quality Table and Headmap
    updateChannelTable();
    updateHeadmapVisuals();

    // BIDS recording counter update
    if (state.recording) {
        state.recordSamples += 4;
        const elapsed = (Date.now() - state.recordStartTime) / 1000;
        const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const ss = String(Math.floor(elapsed % 60)).padStart(2, '0');
        document.getElementById('recTimeElapsed').textContent = `${mm}:${ss}`;
        document.getElementById('recSampleCount').textContent = `${state.recordSamples.toLocaleString()} samples`;
    }
}

function updateChannelTable() {
    const tbody = document.getElementById('channelTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    GPYPE_32_LAYOUT.forEach(ch => {
        const info = state.channels[ch.name];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${ch.name}</strong></td>
            <td>${ch.region}</td>
            <td>${info.rms.toFixed(1)} µV</td>
            <td>${info.ptp.toFixed(1)} µV</td>
            <td><span class="channel-status-badge badge-${info.status}">${info.status.toUpperCase()}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

// ── 7. LIVE TELEMETRY INGESTION & DATA BINDING ──
function applyLiveTelemetry(data) {
    if (!data) return;
    state.connected = data.connected;
    state.isSimulated = !data.connected;
    state.battery = data.battery_pct !== undefined ? data.battery_pct : 80.0;
    state.medianRms = data.median_rms_uv || 20.0;
    state.retouchPins = data.retouch_pins || [];
    state.flatPins = data.flat_pins || [];

    // Battery UI
    const batEl = document.getElementById('batteryLevelText');
    if (batEl) batEl.textContent = `${Math.round(state.battery)}%`;

    // Channels
    if (data.channels) {
        Object.keys(data.channels).forEach(name => {
            if (state.channels[name]) {
                state.channels[name].status = data.channels[name].status;
                state.channels[name].rms = data.channels[name].rms;
                state.channels[name].ptp = data.channels[name].ptp;
            }
        });
    }

    // Scope Traces
    if (data.scope_traces) {
        ['Cz', 'C3', 'C4', 'Oz'].forEach(k => {
            if (data.scope_traces[k] && data.scope_traces[k].length > 0) {
                const arr = data.scope_traces[k];
                const len = Math.min(arr.length, state.scopeSamples);
                for (let i = 0; i < len; i++) {
                    state.scopeData[k][state.scopeSamples - len + i] = arr[i];
                }
            }
        });
    }

    // Neurofeedback SMR Drive & Bank
    if (data.neurofeedback) {
        state.mentalDrive = data.neurofeedback.drive_pct || 50;
        state.lateralization = data.neurofeedback.lateralization || 0.0;
        const driveEl = document.getElementById('hudDriveFill');
        const driveTxt = document.getElementById('hudDriveText');
        if (driveEl) driveEl.style.width = `${state.mentalDrive}%`;
        if (driveTxt) driveTxt.textContent = `Drive: ${state.mentalDrive}%`;

        const steerEl = document.getElementById('hudSteeringText');
        if (steerEl) {
            if (state.lateralization > 0.25) {
                steerEl.textContent = 'BANK RIGHT →';
                steerEl.style.color = '#58A6FF';
            } else if (state.lateralization < -0.25) {
                steerEl.textContent = '← BANK LEFT';
                steerEl.style.color = '#58A6FF';
            } else {
                steerEl.textContent = 'CENTERED';
                steerEl.style.color = '#3FB950';
            }
        }
    }

    updateChannelTable();
    updateHeadmapVisuals();
    if (state.connected) {
        updateConnectionStatus(`LIVE LSL (${data.clean_count !== undefined ? data.clean_count : 32}/32 OK)`, "clean");
    }
}

// Dual HTTP Polling & WebSocket Connector
let ws = null;
function pollTelemetryApi() {
    fetch('/api/telemetry')
        .then(res => res.json())
        .then(data => {
            if (data && data.connected) {
                applyLiveTelemetry(data);
            }
        })
        .catch(() => {});
}

function connectWebSocket() {
    // Start continuous REST polling at 10 FPS
    setInterval(pollTelemetryApi, 100);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            state.connected = true;
            state.isSimulated = false;
            updateConnectionStatus("LIVE LSL STREAM", "clean");
            showToast("Connected to Nautilus LSL Server!");
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                applyLiveTelemetry(data);
            } catch(e) {}
        };

        ws.onclose = () => {
            setTimeout(connectWebSocket, 5000);
        };
    } catch (e) {}
}

function updateConnectionStatus(text, type) {
    const pill = document.getElementById('connectionStatusPill');
    const label = document.getElementById('connectionStatusText');
    if (!pill || !label) return;
    label.textContent = text;
    pill.className = `status-pill ${type}`;
}

// ── 8. BIDS RECORDING CONTROLS ──
function toggleBidsRecording() {
    const btn = document.getElementById('btnToggleBidsRecord');
    const floatBtn = document.getElementById('floatRecordBtn');

    if (!state.recording) {
        state.recording = true;
        state.recordStartTime = Date.now();
        state.recordSamples = 0;
        state.recordEvents = 0;

        btn.textContent = "⏹ Stop & Export BIDS Dataset";
        btn.className = "big-touch-btn btn-record-stop";
        if (floatBtn) {
            floatBtn.textContent = "⏹ Stop Recording";
            floatBtn.className = "big-touch-btn btn-record-stop";
        }

        document.getElementById('recStatusIndicator').textContent = "RECORDING MULTIMODAL";
        document.getElementById('recStatusIndicator').style.color = "#FF7B72";
        showToast("🔴 BIDS Recording Started!");

        // If backend connected, call REST API
        if (state.connected) {
            const sub = document.getElementById('inputSubId').value;
            const ses = document.getElementById('inputSesId').value;
            const task = document.getElementById('selectTaskName').value;
            fetch('/api/bids/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject: sub, session: ses, task: task })
            }).catch(() => {});
        }
    } else {
        state.recording = false;
        btn.textContent = "🔴 Start BIDS Recording";
        btn.className = "big-touch-btn btn-record-start";
        if (floatBtn) {
            floatBtn.textContent = "🔴 Start BIDS Recording";
            floatBtn.className = "big-touch-btn btn-record-start";
        }

        document.getElementById('recStatusIndicator').textContent = "STANDBY";
        document.getElementById('recStatusIndicator').style.color = "#8B949E";
        showToast("⏹ BIDS Dataset Exported to Disk!");

        if (state.connected) {
            fetch('/api/bids/stop', { method: 'POST' }).catch(() => {});
        }
    }
}

// ── 9. TOAST NOTIFICATIONS & UI TABS ──
function showToast(msg) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 250);
    }, 3200);
}

function initTabNavigation() {
    const buttons = document.querySelectorAll('.nav-tab-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            const targetPane = document.getElementById(`tabPane-${tabId}`);
            if (targetPane) targetPane.classList.add('active');

            state.activeTab = tabId;
            if (tabId === 'oscilloscope') resizeScopeCanvas();
        });
    });
}

// ── 10. SYSTEM INITIALIZATION ──
document.addEventListener('DOMContentLoaded', () => {
    initSvgHeadmaps();
    initScopeCanvas();
    initThreeJsTask();
    initTabNavigation();

    // Hook Record Buttons
    const btnRecord = document.getElementById('btnToggleBidsRecord');
    if (btnRecord) btnRecord.addEventListener('click', toggleBidsRecording);

    const floatRecord = document.getElementById('floatRecordBtn');
    if (floatRecord) floatRecord.addEventListener('click', toggleBidsRecording);

    // Start Simulation Loop & Attempt Live WebSocket
    setInterval(runSimulatedTelemetry, 100);
    renderScope();
    connectWebSocket();
});
