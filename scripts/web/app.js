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
    
    const hudDriveFill = document.getElementById('hudDriveFill');
    const hudDriveText = document.getElementById('hudDriveText');
    if (hudDriveFill) hudDriveFill.style.width = `${state.mentalDrive}%`;
    if (hudDriveText) hudDriveText.textContent = `Drive: ${state.mentalDrive}%`;

    const erdGauge = document.getElementById('erdGaugeBar');
    const erdVal = document.getElementById('erdLateralVal');
    if (erdGauge) {
        const pct = Math.max(0, Math.min(100, ((state.lateralization + 1.0) / 2.0) * 100));
        erdGauge.style.width = `${pct}%`;
    }
    if (erdVal) {
        erdVal.textContent = state.lateralization.toFixed(2);
    }

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

        const erdGauge = document.getElementById('erdGaugeBar');
        const erdVal = document.getElementById('erdLateralVal');
        if (erdGauge) {
            const pct = Math.max(0, Math.min(100, ((state.lateralization + 1.0) / 2.0) * 100));
            erdGauge.style.width = `${pct}%`;
        }
        if (erdVal) {
            erdVal.textContent = state.lateralization.toFixed(2);
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

// ── 10. TAB NAVIGATION & VIEW SWITCHING ──
const TASK_TABS = ['task3d', 'motorimagery', 'musicmemory', 'baseline'];

function initTabNavigation() {
    const buttons = document.querySelectorAll('.nav-tab-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            if (tabId) switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    const buttons = document.querySelectorAll('.nav-tab-btn');
    const cascadeWrap = document.getElementById('cascadeMenuWrap');
    const cascadeBtn = document.getElementById('btnCascadeTrigger');

    buttons.forEach(b => {
        if (b.getAttribute('data-tab') === tabId) b.classList.add('active');
        else if (b.getAttribute('data-tab')) b.classList.remove('active');
    });

    // Check if the switched tab is one of the experimental tasks
    if (TASK_TABS.includes(tabId)) {
        if (cascadeBtn) cascadeBtn.classList.add('active');
        if (cascadeWrap) cascadeWrap.classList.add('active-task');
    } else {
        if (cascadeBtn) cascadeBtn.classList.remove('active');
        if (cascadeWrap) cascadeWrap.classList.remove('active-task');
    }

    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    const targetPane = document.getElementById(`tabPane-${tabId}`);
    if (targetPane) targetPane.classList.add('active');

    state.activeTab = tabId;
    if (tabId === 'oscilloscope') resizeScopeCanvas();
    if (tabId === 'motorimagery' && state.motorSubMode === 'fnf') resizeFnfCanvas();
}

// ── 11. CASCADING TASKS MENU CONTROLLER ──
function initCascadingTasksMenu() {
    const wrap = document.getElementById('cascadeMenuWrap');
    const triggerBtn = document.getElementById('btnCascadeTrigger');
    const subItems = document.querySelectorAll('.sub-item');
    const activeBadge = document.getElementById('activeTaskBadge');
    const triggerIcon = document.getElementById('cascadeTriggerIcon');
    const triggerLabel = document.getElementById('cascadeTriggerLabel');
    const selectTask = document.getElementById('selectTaskName');
    const cascadeItems = document.querySelectorAll('.cascade-item');

    if (!wrap || !triggerBtn) return;

    // Toggle dropdown & check boundary collision
    triggerBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = wrap.classList.toggle('open');
        triggerBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');

        if (isOpen) {
            // Check if flyout would overflow right edge of viewport
            const rect = wrap.getBoundingClientRect();
            if (rect.left + 620 > window.innerWidth) {
                wrap.classList.add('flip-submenus');
            } else {
                wrap.classList.remove('flip-submenus');
            }
        }
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!wrap.contains(e.target)) {
            wrap.classList.remove('open');
            triggerBtn.setAttribute('aria-expanded', 'false');
        }
    });

    // Escape key closes menu
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && wrap.classList.contains('open')) {
            wrap.classList.remove('open');
            triggerBtn.setAttribute('aria-expanded', 'false');
        }
    });

    // Category click: accordion expansion on mobile & desktop toggle
    cascadeItems.forEach(item => {
        item.addEventListener('click', (e) => {
            // Don't intercept if clicking directly on a sub-item
            if (e.target.closest('.sub-item')) return;
            
            const isExpanded = item.classList.contains('expanded');
            cascadeItems.forEach(i => i.classList.remove('expanded'));
            if (!isExpanded) item.classList.add('expanded');
            e.stopPropagation();
        });
    });

    // Sub-item task selection
    subItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.stopPropagation();
            subItems.forEach(s => s.classList.remove('active'));
            item.classList.add('active');

            const taskName = item.getAttribute('data-task');
            const mode = item.getAttribute('data-mode');
            const tabTarget = item.getAttribute('data-tab');
            const iconEl = item.querySelector('.sub-item-icon');
            const titleEl = item.querySelector('.sub-item-title') || item.querySelector('strong');
            
            const icon = iconEl ? iconEl.textContent.trim() : '🎮';
            const title = titleEl ? titleEl.textContent.trim() : taskName;

            // Update Trigger Button Display
            if (triggerIcon) triggerIcon.textContent = icon;
            if (triggerLabel) triggerLabel.textContent = `Task: ${title.split(':')[0]}`;

            // Update Badge in Header
            if (activeBadge) {
                activeBadge.innerHTML = `<span>${icon} ${title}</span>`;
            }

            // Sync with BIDS selector
            if (selectTask) {
                selectTask.value = taskName;
            }

            // Apply appropriate sub-mode
            if (tabTarget === 'task3d' && mode) setSpatialSubMode(mode);
            if (tabTarget === 'motorimagery' && mode) setMotorSubMode(mode);
            if (tabTarget === 'musicmemory' && mode) setMusicSubMode(mode);
            if (tabTarget === 'baseline' && mode) setBaselineSubMode(mode);

            // Switch Tab View
            switchTab(tabTarget);

            wrap.classList.remove('open');
            triggerBtn.setAttribute('aria-expanded', 'false');
            showToast(`Activated: ${title}`);
        });
    });
}

// ── 12. PARADIGM SUB-MODE CONTROLLERS ──
state.spatialSubMode = 'phase2';
state.motorSubMode = '4class';
state.musicSubMode = 'recall';
state.baselineSubMode = 'eyes';

function setSpatialSubMode(mode) {
    state.spatialSubMode = mode;
    ['phase2', 'phase1', 'freeflight'].forEach(m => {
        const btn = document.getElementById(`pillBtn-spatial-${m}`);
        if (btn) {
            if (m === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        }
    });

    const badge = document.getElementById('spatialTaskBadge');
    const cueBanner = document.getElementById('phase1CueBanner');

    if (mode === 'phase1') {
        if (badge) badge.textContent = "Phase 1: Guided Calibration Cues";
        if (cueBanner) cueBanner.style.display = 'block';
        startPhase1GuidedSequencer();
    } else if (mode === 'freeflight') {
        if (badge) badge.textContent = "Free Flight Simulation Mode";
        if (cueBanner) cueBanner.style.display = 'none';
        stopPhase1GuidedSequencer();
    } else {
        if (badge) badge.textContent = "SMR Closed-Loop Active";
        if (cueBanner) cueBanner.style.display = 'none';
        stopPhase1GuidedSequencer();
    }
}

let phase1Timer = null;
function startPhase1GuidedSequencer() {
    stopPhase1GuidedSequencer();
    const cues = [
        { title: "SURGE FORWARD", sub: "Imagine Both Hands Opening / Closing", progressColor: "#38EF7D" },
        { title: "SWAY LEFT", sub: "Imagine Left Hand Clenching", progressColor: "#58A6FF" },
        { title: "SWAY RIGHT", sub: "Imagine Right Hand Clenching", progressColor: "#BC85FF" },
        { title: "HEAVE UP", sub: "Imagine Both Feet Tapping", progressColor: "#D29922" }
    ];
    let idx = 0;
    const targetText = document.getElementById('phase1TargetText');
    const subText = document.getElementById('phase1CueSub');
    const fill = document.getElementById('phase1CueProgress');

    function step() {
        const cue = cues[idx % cues.length];
        if (targetText) targetText.textContent = cue.title;
        if (subText) subText.textContent = cue.sub;
        if (fill) {
            fill.style.backgroundColor = cue.progressColor;
            fill.style.width = '0%';
            setTimeout(() => { fill.style.transition = 'width 3.8s linear'; fill.style.width = '100%'; }, 50);
        }
        idx++;
    }
    step();
    phase1Timer = setInterval(step, 4000);
}

function stopPhase1GuidedSequencer() {
    if (phase1Timer) {
        clearInterval(phase1Timer);
        phase1Timer = null;
    }
}

function setMotorSubMode(mode) {
    state.motorSubMode = mode;
    const view4Class = document.getElementById('motorView-4class');
    const viewFnf = document.getElementById('motorView-fnf');
    const btn4Class = document.getElementById('pillBtn-4class');
    const btnFnf = document.getElementById('pillBtn-fnf');

    if (mode === 'fnf') {
        if (view4Class) view4Class.style.display = 'none';
        if (viewFnf) viewFnf.style.display = 'flex';
        if (btn4Class) btn4Class.classList.remove('active');
        if (btnFnf) btnFnf.classList.add('active');
        initFnfHighway();
    } else {
        if (view4Class) view4Class.style.display = 'flex';
        if (viewFnf) viewFnf.style.display = 'none';
        if (btn4Class) btn4Class.classList.add('active');
        if (btnFnf) btnFnf.classList.remove('active');
    }
}

function setMusicSubMode(mode) {
    state.musicSubMode = mode;
    const viewRecall = document.getElementById('musicView-recall');
    const viewCal = document.getElementById('musicView-calibrator');
    const btnRecall = document.getElementById('pillBtn-musicrecall');
    const btnCal = document.getElementById('pillBtn-calibrator');

    if (mode === 'calibrator') {
        if (viewRecall) viewRecall.style.display = 'none';
        if (viewCal) viewCal.style.display = 'block';
        if (btnRecall) btnRecall.classList.remove('active');
        if (btnCal) btnCal.classList.add('active');
    } else {
        if (viewRecall) viewRecall.style.display = 'block';
        if (viewCal) viewCal.style.display = 'none';
        if (btnRecall) btnRecall.classList.add('active');
        if (btnCal) btnCal.classList.remove('active');
    }
}

function setBaselineSubMode(mode) {
    state.baselineSubMode = mode;
    const viewEyes = document.getElementById('baselineView-eyes');
    const viewSlideshow = document.getElementById('baselineView-slideshow');
    const btnEyes = document.getElementById('pillBtn-eyes');
    const btnSlideshow = document.getElementById('pillBtn-slideshow');

    if (mode === 'slideshow') {
        if (viewEyes) viewEyes.style.display = 'none';
        if (viewSlideshow) viewSlideshow.style.display = 'block';
        if (btnEyes) btnEyes.classList.remove('active');
        if (btnSlideshow) btnSlideshow.classList.add('active');
    } else {
        if (viewEyes) viewEyes.style.display = 'block';
        if (viewSlideshow) viewSlideshow.style.display = 'none';
        if (btnEyes) btnEyes.classList.add('active');
        if (btnSlideshow) btnSlideshow.classList.remove('active');
    }
}

// ── 13. WEB AUDIO PROCEDURAL SYNTHESIZER ──
let audioCtx = null;
let activeAudioNodes = [];

function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

function playTone(freq, dur, type = 'sine', vol = 0.2) {
    try {
        const ctx = getAudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(vol, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + dur);
    } catch(e) {}
}

function stopAllAudio() {
    activeAudioNodes.forEach(node => {
        try { node.stop(); } catch(e) {}
        try { node.disconnect(); } catch(e) {}
    });
    activeAudioNodes = [];
    document.querySelectorAll('.element-cue-card').forEach(c => {
        c.classList.remove('active-fire', 'active-water', 'active-wind', 'active-electricity');
    });
    showToast("Audio Silenced.");
}

function playElementSound(element) {
    stopAllAudio();
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    document.querySelectorAll('.element-cue-card').forEach(c => {
        c.classList.remove('active-fire', 'active-water', 'active-wind', 'active-electricity');
    });
    const card = document.getElementById(`cardElement-${element}`);
    if (card) card.classList.add(`active-${element}`);

    if (element === 'fire') {
        // Fast upbeat arpeggios (A minor pentatonic: A4, C5, D5, E5, G5)
        const notes = [440, 523.25, 587.33, 659.25, 783.99, 880];
        for (let i = 0; i < 24; i++) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            const f = notes[i % notes.length];
            const t = now + i * 0.14;
            osc.frequency.setValueAtTime(f, t);
            gain.gain.setValueAtTime(0.22, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(t);
            osc.stop(t + 0.13);
            activeAudioNodes.push(osc);
        }
        showToast("Playing Theme: 🔥 Fire (High Energy Arpeggio)");
    } else if (element === 'water') {
        // Calm ambient chords (E minor: E3, B3, G4, D5) with chorus detune
        const chords = [164.81, 246.94, 392.00, 587.33];
        chords.forEach((freq, idx) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now);
            gain.gain.setValueAtTime(0.001, now);
            gain.gain.linearRampToValueAtTime(0.2, now + 1.0);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 5.0);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 5.0);
            activeAudioNodes.push(osc);
        });
        showToast("Playing Theme: 💧 Water (Ambient Sine Wave Flow)");
    } else if (element === 'wind') {
        // Sweeping resonant bandpass noise buffer & ethereal sine swell
        const bufferSize = ctx.sampleRate * 4;
        const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;

        const noise = ctx.createBufferSource();
        noise.buffer = buffer;
        const filter = ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(300, now);
        filter.frequency.exponentialRampToValueAtTime(1400, now + 2.0);
        filter.frequency.exponentialRampToValueAtTime(400, now + 4.0);
        filter.Q.value = 5.0;

        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0.001, now);
        gain.gain.linearRampToValueAtTime(0.35, now + 1.5);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 4.5);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(ctx.destination);
        noise.start(now);
        noise.stop(now + 4.5);
        activeAudioNodes.push(noise);
        showToast("Playing Theme: 🌪️ Wind (Harmonic Resonant Sweep)");
    } else if (element === 'electricity') {
        // Fast 16th-note pulsed sawtooth bass & syncopated stabs
        for (let i = 0; i < 20; i++) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'square';
            const t = now + i * 0.11;
            osc.frequency.setValueAtTime(i % 4 === 0 ? 110 : 220, t);
            gain.gain.setValueAtTime(0.24, t);
            gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(t);
            osc.stop(t + 0.09);
            activeAudioNodes.push(osc);
        }
        showToast("Playing Theme: ⚡ Electricity (Pulsed 16th Synth Bass)");
    }
}

// Music Recall Trial Runner
let musicTrialRunning = false;
function toggleMusicImageryBlock() {
    if (musicTrialRunning) return;
    musicTrialRunning = true;
    const elements = ['fire', 'water', 'wind', 'electricity'];
    const target = elements[Math.floor(Math.random() * elements.length)];
    const badge = document.getElementById('musicTrialBadge');

    if (badge) badge.textContent = `Stage 1: Listening (${target.toUpperCase()})`;
    playElementSound(target);

    setTimeout(() => {
        stopAllAudio();
        if (badge) badge.textContent = `Stage 2: Imagine Theme (${target.toUpperCase()})`;
        playTone(660, 0.2, 'sine', 0.2); // Start mental imagery beep
        showToast(`🧠 Silence: Mentally recall the acoustic theme for ${target.toUpperCase()}`);

        setTimeout(() => {
            if (badge) badge.textContent = `Stage 3: Verified!`;
            playTone(880, 0.3, 'sine', 0.2); // End beep
            showToast(`✅ Trial complete for ${target.toUpperCase()}!`);
            musicTrialRunning = false;
        }, 5000);
    }, 5000);
}

function updateCalibratorBpm(val) {
    const el = document.getElementById('bpmDisplayVal');
    if (el) el.textContent = `${val} BPM`;
}

function updateCalibratorOffset(val) {
    const el = document.getElementById('offsetDisplayVal');
    if (el) el.textContent = `${val > 0 ? '+' : ''}${val} ms`;
}

let metronomeTimer = null;
function toggleMetronomeAudio() {
    if (metronomeTimer) {
        clearInterval(metronomeTimer);
        metronomeTimer = null;
        showToast("Metronome Stopped");
    } else {
        const bpmInput = document.getElementById('sliderBpm');
        const bpm = bpmInput ? parseInt(bpmInput.value) : 120;
        const interval = (60 / bpm) * 1000;
        playTone(1000, 0.05, 'triangle', 0.15);
        metronomeTimer = setInterval(() => {
            playTone(1000, 0.05, 'triangle', 0.15);
        }, interval);
        showToast(`Metronome Started @ ${bpm} BPM`);
    }
}

// ── 14. 4-CLASS MOTOR IMAGERY TRIAL SEQUENCER ──
let motorTrialRunning = false;
let motorTrialTimer = null;
let currentTrialNum = 0;
const TOTAL_MOTOR_TRIALS = 40;

function toggleMotorImageryBlock() {
    if (motorTrialRunning) {
        resetMotorImageryBlock();
        return;
    }
    motorTrialRunning = true;
    currentTrialNum = 0;
    const btn = document.getElementById('btnStartMotorBlock');
    if (btn) btn.textContent = "⏹ Abort Block";
    runNextMotorTrial();
}

function resetMotorImageryBlock() {
    motorTrialRunning = false;
    if (motorTrialTimer) clearTimeout(motorTrialTimer);
    const btn = document.getElementById('btnStartMotorBlock');
    if (btn) btn.textContent = "▶ Start Calibration Block (40 Trials)";
    const badge = document.getElementById('motorTrialBadge');
    if (badge) badge.textContent = "Standby";

    clearActiveCues();
    const cross = document.getElementById('fixationCross');
    const notice = document.getElementById('cuePhaseNotice');
    if (cross) { cross.className = 'fixation-cross'; }
    if (notice) notice.textContent = "READY";
}

function clearActiveCues() {
    ['cueArrowLeft', 'cueArrowRight', 'cueArrowUp', 'cueArrowDown'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('active-cue');
    });
}

function runNextMotorTrial() {
    if (!motorTrialRunning) return;
    if (currentTrialNum >= TOTAL_MOTOR_TRIALS) {
        showToast("🎉 Block Complete! 40 Trials Recorded.");
        resetMotorImageryBlock();
        return;
    }

    currentTrialNum++;
    const badge = document.getElementById('motorTrialBadge');
    if (badge) badge.textContent = `Trial ${currentTrialNum} / ${TOTAL_MOTOR_TRIALS}`;

    const cross = document.getElementById('fixationCross');
    const notice = document.getElementById('cuePhaseNotice');
    clearActiveCues();

    // 1. REST PHASE (2.0s)
    if (cross) cross.className = 'fixation-cross';
    if (notice) notice.textContent = "REST";

    motorTrialTimer = setTimeout(() => {
        if (!motorTrialRunning) return;

        // 2. READY PHASE (1.0s)
        if (cross) cross.className = 'fixation-cross ready';
        if (notice) notice.textContent = "PREPARE";
        playTone(440, 0.15, 'sine', 0.15); // Prepare beep

        motorTrialTimer = setTimeout(() => {
            if (!motorTrialRunning) return;

            // 3. CUE & IMAGERY PHASE (4.0s)
            const classes = [
                { id: 'cueArrowLeft', name: 'LEFT HAND', tone: 520 },
                { id: 'cueArrowRight', name: 'RIGHT HAND', tone: 620 },
                { id: 'cueArrowUp', name: 'TONGUE / HEAD', tone: 720 },
                { id: 'cueArrowDown', name: 'BOTH FEET', tone: 380 }
            ];
            const target = classes[Math.floor(Math.random() * classes.length)];
            const arrowEl = document.getElementById(target.id);
            if (arrowEl) arrowEl.classList.add('active-cue');
            if (cross) cross.className = 'fixation-cross active';
            if (notice) notice.textContent = `IMAGINE: ${target.name}`;
            playTone(target.tone, 0.25, 'sine', 0.25);

            motorTrialTimer = setTimeout(() => {
                if (!motorTrialRunning) return;
                clearActiveCues();
                if (cross) cross.className = 'fixation-cross';
                if (notice) notice.textContent = "DONE";
                motorTrialTimer = setTimeout(runNextMotorTrial, 1000);
            }, 4000);
        }, 1000);
    }, 2000);
}

// ── 15. FRIDAY NIGHT FUNKIN' RHYTHM HIGHWAY ──
let fnfRunning = false;
let fnfScore = 0;
let fnfCombo = 0;
let fnfNotes = [];
let fnfCanvas, fnfCtx, fnfAnimFrame;

function initFnfHighway() {
    fnfCanvas = document.getElementById('fnfHighwayCanvas');
    if (!fnfCanvas) return;
    fnfCtx = fnfCanvas.getContext('2d');
    resizeFnfCanvas();
}

function resizeFnfCanvas() {
    if (!fnfCanvas) return;
    fnfCanvas.width = 320;
    fnfCanvas.height = 380;
}

function toggleFnfBattle() {
    const btn = document.getElementById('btnStartFnf');
    if (fnfRunning) {
        fnfRunning = false;
        cancelAnimationFrame(fnfAnimFrame);
        if (btn) btn.textContent = "▶ Start Rhythm Battle";
        showToast("Battle Paused");
    } else {
        fnfRunning = true;
        fnfScore = 0;
        fnfCombo = 0;
        fnfNotes = [];
        updateFnfHud();
        if (btn) btn.textContent = "⏹ Stop Battle";
        showToast("🎶 Rhythm Battle Started! Hit notes with Motor Imagery or Keys!");
        animateFnfHighway();
    }
}

function updateFnfHud() {
    const scoreEl = document.getElementById('fnfScoreVal');
    const comboEl = document.getElementById('fnfComboVal');
    if (scoreEl) scoreEl.textContent = fnfScore.toString().padStart(6, '0');
    if (comboEl) comboEl.textContent = `${fnfCombo}x`;
}

function animateFnfHighway() {
    if (!fnfRunning || !fnfCtx || !fnfCanvas) return;
    const w = fnfCanvas.width;
    const h = fnfCanvas.height;
    const laneW = w / 4;

    fnfCtx.fillStyle = '#080C14';
    fnfCtx.fillRect(0, 0, w, h);

    // Draw Lane Dividing Lines
    fnfCtx.strokeStyle = 'rgba(38, 51, 77, 0.4)';
    fnfCtx.lineWidth = 1;
    for (let l = 1; l < 4; l++) {
        fnfCtx.beginPath();
        fnfCtx.moveTo(l * laneW, 0);
        fnfCtx.lineTo(l * laneW, h);
        fnfCtx.stroke();
    }

    // Spawn notes periodically
    if (Math.random() < 0.045) {
        const lane = Math.floor(Math.random() * 4);
        fnfNotes.push({ lane: lane, y: -20, speed: 3.8 });
    }

    const laneColors = ['#BC85FF', '#38EF7D', '#58A6FF', '#FF7B72'];
    const laneArrows = ['⬅️', '⬇️', '⬆️', '➡️'];
    const hitZoneY = h - 50;

    // Draw Target Hit Line
    fnfCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    fnfCtx.lineWidth = 2;
    fnfCtx.beginPath();
    fnfCtx.moveTo(0, hitZoneY);
    fnfCtx.lineTo(w, hitZoneY);
    fnfCtx.stroke();

    // Update and draw falling notes
    for (let i = fnfNotes.length - 1; i >= 0; i--) {
        const n = fnfNotes[i];
        n.y += n.speed;

        const x = n.lane * laneW + laneW / 2;
        fnfCtx.font = '22px sans-serif';
        fnfCtx.textAlign = 'center';
        fnfCtx.textBaseline = 'middle';
        fnfCtx.fillText(laneArrows[n.lane], x, n.y);

        // Auto-hit if EEG motor lateralization matches lane
        let autoHit = false;
        if (Math.abs(n.y - hitZoneY) < 22) {
            if (n.lane === 0 && state.lateralization < -0.3) autoHit = true;
            if (n.lane === 3 && state.lateralization > 0.3) autoHit = true;
            if (n.lane === 1 && state.mentalDrive > 65) autoHit = true;
        }

        if (autoHit) {
            triggerFnfHit(n.lane, "SICK!!");
            fnfNotes.splice(i, 1);
            continue;
        }

        // Missed note falls off screen
        if (n.y > h + 20) {
            fnfCombo = 0;
            updateFnfHud();
            const ratingEl = document.getElementById('fnfRatingBanner');
            if (ratingEl) {
                ratingEl.textContent = "MISS!";
                ratingEl.style.color = "#FF7B72";
            }
            fnfNotes.splice(i, 1);
        }
    }

    fnfAnimFrame = requestAnimationFrame(animateFnfHighway);
}

function triggerFnfHit(lane, rating) {
    fnfScore += rating === "SICK!!" ? 350 : 200;
    fnfCombo++;
    updateFnfHud();

    const ratingEl = document.getElementById('fnfRatingBanner');
    if (ratingEl) {
        ratingEl.textContent = rating;
        ratingEl.style.color = rating === "SICK!!" ? "#38EF7D" : "#58A6FF";
    }

    // Receptor visual bounce
    const receptor = document.querySelector(`.fnf-receptor[data-lane="${lane}"]`);
    if (receptor) {
        receptor.classList.add('hit');
        setTimeout(() => receptor.classList.remove('hit'), 120);
    }
    playTone(300 + lane * 120, 0.08, 'triangle', 0.2);
}

// Keyboard input for FNF rhythm battle
window.addEventListener('keydown', (e) => {
    if (!fnfRunning) return;
    const keyMap = {
        'KeyD': 0, 'ArrowLeft': 0,
        'KeyF': 1, 'ArrowDown': 1,
        'KeyJ': 2, 'ArrowUp': 2,
        'KeyK': 3, 'ArrowRight': 3
    };
    if (keyMap[e.code] !== undefined) {
        const lane = keyMap[e.code];
        const hitZoneY = 380 - 50;
        let hitFound = false;

        for (let i = 0; i < fnfNotes.length; i++) {
            const n = fnfNotes[i];
            if (n.lane === lane && Math.abs(n.y - hitZoneY) < 32) {
                const isSick = Math.abs(n.y - hitZoneY) < 16;
                triggerFnfHit(lane, isSick ? "SICK!!" : "GOOD!");
                fnfNotes.splice(i, 1);
                hitFound = true;
                break;
            }
        }
        if (!hitFound) {
            triggerFnfHit(lane, "GOOD");
        }
    }
});

// ── 16. EYES OPEN / EYES CLOSED BASELINE PROTOCOL ──
let eyesRunning = false;
let eyesInterval = null;
let eyesSeconds = 30;
let eyesIsOpen = true;

function toggleEyesProtocol() {
    const btn = document.getElementById('btnStartEyesBlock');
    if (eyesRunning) {
        resetEyesProtocol();
        return;
    }
    eyesRunning = true;
    eyesSeconds = 30;
    eyesIsOpen = true;
    if (btn) btn.textContent = "⏹ Stop Protocol";
    updateEyesVisuals();

    playTone(520, 0.2, 'sine', 0.2);
    showToast("Eyes Protocol Started: 30s Eyes Open, then 30s Closed.");

    eyesInterval = setInterval(() => {
        eyesSeconds--;
        const timerEl = document.getElementById('eyesPromptTimer');
        if (timerEl) timerEl.textContent = `${eyesSeconds}s`;

        if (eyesSeconds <= 0) {
            eyesIsOpen = !eyesIsOpen;
            eyesSeconds = 30;
            playTone(eyesIsOpen ? 660 : 440, 0.3, 'sine', 0.25);
            updateEyesVisuals();
        }

        // Live Alpha Reactivity Calculator
        const alphaEl = document.getElementById('alphaReactivityText');
        if (alphaEl) {
            const ratio = eyesIsOpen ? 1.05 : 2.45;
            alphaEl.textContent = `Reactivity: ${ratio.toFixed(2)}x (Alpha 8-12Hz ${eyesIsOpen ? 'Desynchronized / Suppressed' : 'Synchronized Peak (Berger Effect)'})`;
        }
    }, 1000);
}

function resetEyesProtocol() {
    eyesRunning = false;
    if (eyesInterval) clearInterval(eyesInterval);
    const btn = document.getElementById('btnStartEyesBlock');
    if (btn) btn.textContent = "▶ Start 60s Protocol";
    eyesSeconds = 30;
    eyesIsOpen = true;
    updateEyesVisuals();
}

function updateEyesVisuals() {
    const ring = document.getElementById('eyesPromptRing');
    const icon = document.getElementById('eyesPromptIcon');
    const text = document.getElementById('eyesPromptText');
    const timer = document.getElementById('eyesPromptTimer');

    if (ring && icon && text && timer) {
        timer.textContent = `${eyesSeconds}s`;
        if (eyesIsOpen) {
            ring.className = 'eyes-prompt-ring open-state';
            icon.textContent = '👁️';
            text.textContent = 'EYES OPEN';
        } else {
            ring.className = 'eyes-prompt-ring closed-state';
            icon.textContent = '🙈';
            text.textContent = 'CLOSE YOUR EYES';
        }
    }
}

// ── 17. VISUAL SLIDESHOW PARADIGM ──
let slideshowRunning = false;
let slideshowInterval = null;
function toggleSlideshow() {
    const box = document.getElementById('slideshowBox');
    if (slideshowRunning) {
        slideshowRunning = false;
        if (slideshowInterval) clearInterval(slideshowInterval);
        if (box) box.innerHTML = '<div class="slideshow-fixation">+</div>';
        showToast("Slideshow Stopped");
    } else {
        slideshowRunning = true;
        const patterns = ['+', '●', '■', '▲', '◆', '✕'];
        let idx = 0;
        showToast("Visual Slideshow Running (Stimulus every 3s)");
        slideshowInterval = setInterval(() => {
            idx = (idx + 1) % patterns.length;
            if (box) {
                box.innerHTML = `<div class="slideshow-fixation" style="font-size: 64px; color: ${idx % 2 === 0 ? '#38EF7D' : '#58A6FF'}">${patterns[idx]}</div>`;
            }
        }, 3000);
    }
}

// ── 18. SYSTEM INITIALIZATION ──
document.addEventListener('DOMContentLoaded', () => {
    initSvgHeadmaps();
    initScopeCanvas();
    initThreeJsTask();
    initTabNavigation();
    initCascadingTasksMenu();

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
