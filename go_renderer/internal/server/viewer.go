package server

import "fmt"

func getViewerHTML(sessionID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Medical 3D Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
#canvas-container { width: 100vw; height: 100vh; }
canvas { display: block; }
#controls {
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,0.7); padding: 12px; border-radius: 8px;
    color: #fff; font-size: 13px; min-width: 180px;
}
#controls label { display: block; margin: 6px 0 2px; }
#controls input[type=range] { width: 100%%; }
.label-toggle { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.label-toggle input { accent-color: #0078D4; }
#loading {
    position: absolute; top: 50%%; left: 50%%; transform: translate(-50%%,-50%%);
    color: #fff; font-size: 18px; display: none;
}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="controls">
    <strong>Controls</strong>
    <label>Opacity</label>
    <input type="range" id="opacity" min="0" max="1" step="0.05" value="0.6">
    <div id="label-toggles"></div>
    <label>Background</label>
    <input type="color" id="bg-color" value="#1a1a2e">
</div>
<div id="loading">Loading meshes...</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const SESSION_ID = "%s";
const API_BASE = window.location.origin;
const LABEL_COLORS = {1:0xFF4444,2:0x44FF44,3:0x4444FF,4:0xFFFF44,5:0xFF44FF,6:0x44FFFF,7:0xFF8800,8:0x8800FF};
const LABEL_NAMES = {1:'Aorta',2:'Left Ventricle',3:'Valve Region',4:'Label 4',5:'Label 5',6:'Label 6',7:'Label 7',8:'Label 8'};
let scene, camera, renderer, controls;
let meshes = {};

function init() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 10000);
    camera.position.set(0, 0, 300);
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    scene.add(new THREE.AmbientLight(0x404040, 0.6));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(1,1,1);
    scene.add(dl);
    const bl = new THREE.DirectionalLight(0xffffff, 0.3);
    bl.position.set(-1,-1,-1);
    scene.add(bl);
    window.addEventListener('resize', onResize);
    loadSession();
    animate();
}

async function loadSession() {
    document.getElementById('loading').style.display = 'block';
    try {
        const resp = await fetch(API_BASE+'/api/v1/session/'+SESSION_ID);
        if (!resp.ok) { document.getElementById('loading').textContent='Session not found'; return; }
        const data = await resp.json();
        if (data.labels && data.labels.length > 0) {
            for (const label of data.labels) await loadMesh(label);
            centerCamera();
        } else {
            document.getElementById('loading').textContent='No meshes. Run segmentation first.';
            return;
        }
    } catch(err) { document.getElementById('loading').textContent='Error: '+err.message; return; }
    document.getElementById('loading').style.display = 'none';
}

async function loadMesh(label) {
    const resp = await fetch(API_BASE+'/api/v1/mesh/'+SESSION_ID+'/'+label);
    if (!resp.ok) return;
    const buffer = await resp.arrayBuffer();
    const view = new DataView(buffer);
    let offset = 0;
    const numVerts = view.getUint32(offset, true); offset += 4;
    const numTris = view.getUint32(offset, true); offset += 4;
    const positions = new Float32Array(numVerts * 3);
    for (let i = 0; i < numVerts*3; i++) { positions[i] = view.getFloat32(offset, true); offset += 4; }
    const indices = new Uint32Array(numTris * 3);
    for (let i = 0; i < numTris*3; i++) { indices[i] = view.getUint32(offset, true); offset += 4; }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    const color = LABEL_COLORS[parseInt(label)] || 0xcccccc;
    const material = new THREE.MeshPhongMaterial({color, transparent:true, opacity:0.6, side:THREE.DoubleSide, depthWrite:false});
    const m = new THREE.Mesh(geometry, material);
    scene.add(m);
    meshes[label] = m;
    const toggleDiv = document.getElementById('label-toggles');
    const name = LABEL_NAMES[parseInt(label)] || 'Label '+label;
    const div = document.createElement('div');
    div.className = 'label-toggle';
    div.innerHTML = '<input type="checkbox" checked data-label="'+label+'"><span style="color:#'+color.toString(16).padStart(6,'0')+'">'+name+'</span>';
    div.querySelector('input').addEventListener('change', function(e) { meshes[label].visible = e.target.checked; });
    toggleDiv.appendChild(div);
}

function centerCamera() {
    const box = new THREE.Box3();
    for (const m of Object.values(meshes)) box.expandByObject(m);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    camera.position.copy(center);
    camera.position.z += Math.max(size.x,size.y,size.z)*1.5;
    controls.target.copy(center);
    controls.update();
}

function onResize() {
    camera.aspect = window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

document.getElementById('opacity').addEventListener('input', function(e) {
    const val = parseFloat(e.target.value);
    for (const m of Object.values(meshes)) m.material.opacity = val;
});
document.getElementById('bg-color').addEventListener('input', function(e) {
    scene.background = new THREE.Color(e.target.value);
});
init();
</script>
</body>
</html>`, sessionID)
}
