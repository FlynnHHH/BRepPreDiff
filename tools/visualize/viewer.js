import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const MODEL_FILES = [];
const grid = document.querySelector("#modelGrid");
const metricsSummary = document.querySelector("#metricsSummary");
const searchInput = document.querySelector("#searchInput");
const datasetFilter = document.querySelector("#datasetFilter");
const favoritesOnly = document.querySelector("#favoritesOnly");
const loader = new PLYLoader();
const viewers = new Map();
let activeFiles = MODEL_FILES;
let predictionManifests = new Map();
let manifestSamples = new Map();
let predictionManifestsSignature = null;
let observer = null;
const FAVORITES_KEY = "curvseg-2026-6-22-favorite-samples";
let favorites = loadFavorites();
let serverFavoritesReady = false;

const PRIMITIVE_LEGEND = [
  { id: 0, en: "NonTransition", zh: "非过渡面", rgb: "218, 222, 230" },
  { id: 1, en: "VBF", zh: "VBF 点过渡面", rgb: "255, 79, 163" },
  { id: 2, en: "EBF", zh: "EBF 边过渡面", rgb: "255, 212, 0" }
];

const BINARY_LEGEND = [
  { id: 0, en: "NonTransition", zh: "非过渡面", rgb: "218, 222, 230" },
  { id: 1, en: "Transition", zh: "过渡面（VBF/EBF）", rgb: "255, 79, 163" }
];

const MODEL_BASE_URL = "./results/";
const INSTANCE_SUFFIX = "_instance_pred_rgb.ply";
const SEMANTIC_SUFFIX = "_semantic_pred.ply";
const PREDICTION_MANIFEST_SOURCES = [
  {
    key: "finetune-test",
    filename: "finetune_test_manifest.json",
    metricsTitle: "BRepPreDiff expanded-data default MLP on official test split",
    metricsSubtitle: "三分类 Macro 指标 · NonTransition / VBF / EBF",
    modelName: "BRepPreDiff expanded-data default MLP",
    visualizationNumClasses: 3,
    gtTitle: "BRepPreDiff test split 三分类 GT",
    predictionTitle: "扩充数据默认 MLP · NonTransition / VBF / EBF 预测"
  },
  {
    key: "brepprediff-best-lxy",
    filename: "brepprediff_best_lxy_manifest.json",
    metricsTitle: "BRepPreDiff official-test best MLP on testset_lxy",
    metricsSubtitle: "三分类 Macro 指标 · NonTransition / VBF / EBF",
    modelName: "BRepPreDiff expanded-data best MLP",
    visualizationNumClasses: 3,
    gtTitle: "testset_lxy SEG 三分类 GT",
    predictionTitle: "BRepPreDiff 官方 test 最优 MLP 三分类预测"
  },
  {
    key: "cjq-step-numeric",
    filename: "cjq_step_numeric_manifest.json",
    metricsTitle: "BRepPreDiff best MLP on cjq step_numeric",
    metricsSubtitle: "无 GT，仅展示三分类预测",
    modelName: "BRepPreDiff expanded-data best MLP",
    visualizationNumClasses: 3,
    gtTitle: "cjq step_numeric 原始模型（无 GT）",
    predictionTitle: "BRepPreDiff 官方 test 最优 MLP 三分类预测"
  },
  {
    key: "filletrec",
    filename: "prediction_manifest.json",
    metricsTitle: "BRepPreDiff on FilletRec Test Set",
    modelName: "BRepPreDiff",
    visualizationNumClasses: 2,
    gtTitle: "FilletRec GT 过渡面",
    predictionTitle: "BRepPreDiff 过渡面预测"
  },
  {
    key: "filletrec-model-filletrec-testset",
    filename: "filletrec_on_filletrec_manifest.json",
    metricsTitle: "filletrec on filletrec testset",
    modelName: "FilletRec",
    visualizationNumClasses: 2,
    gtTitle: "FilletRec GT 过渡面",
    predictionTitle: "FilletRec 过渡面预测"
  },
  {
    key: "filletrec-ourtestset",
    filename: "filletrec_ourtestset_manifest.json",
    metricsTitle: "filletrec on ourtestset",
    modelName: "FilletRec",
    visualizationNumClasses: 2,
    gtTitle: "ourtestset GT 过渡面",
    predictionTitle: "FilletRec 过渡面预测"
  }
];

function safeName(name) {
  return encodeURIComponent(name);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}

class SharedCameraController {
  constructor() {
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.001, 100000);
    this.camera.position.set(0, 0, 1);
    this.controls = null;
    this.target = new THREE.Vector3(0, 0, 0);
  }

  attachCanvas(domElement) {
    if (this.controls) {
      this.controls.dispose();
    }
    this.controls = new TrackballControls(this.camera, domElement);
    this.controls.rotateSpeed = 3.0;
    this.controls.zoomSpeed = 1.2;
    this.controls.panSpeed = 0.8;
    this.controls.staticMoving = false;
    this.controls.dynamicDampingFactor = 0.12;
    this.controls.target.copy(this.target);
    this.controls.update();
  }

  syncFrom(other) {
    this.camera.copy(other.camera);
    this.target.copy(other.target);
    if (this.controls) {
      this.controls.target.copy(this.target);
      this.controls.update();
    }
  }

  setTransform(camera, target) {
    this.camera.position.copy(camera.position);
    this.camera.rotation.copy(camera.rotation);
    this.camera.zoom = camera.zoom;
    this.camera.fov = camera.fov;
    this.camera.up.copy(camera.up);
    this.camera.near = camera.near;
    this.camera.far = camera.far;
    this.camera.aspect = camera.aspect;
    this.camera.updateProjectionMatrix();

    this.target.copy(target);
    if (this.controls) {
      this.controls.target.copy(target);
      this.controls.update();
    }
  }
}

class PairCardViewer {
  constructor(card, instanceName, semanticName) {
    this.card = card;
    this.instanceName = instanceName;
    this.semanticName = semanticName;
    this.instanceStatus = card.querySelector(".instance-status");
    this.semanticStatus = card.querySelector(".semantic-status");

    this.instanceCanvas = card.querySelector("canvas.instance-view");
    this.semanticCanvas = card.querySelector("canvas.semantic-view");

    this.instanceRenderer = new THREE.WebGLRenderer({ canvas: this.instanceCanvas, antialias: true });
    this.semanticRenderer = new THREE.WebGLRenderer({ canvas: this.semanticCanvas, antialias: true });

    [this.instanceRenderer, this.semanticRenderer].forEach((renderer) => {
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.7));
      renderer.setClearColor(0xffffff, 1);
    });

    this.instanceScene = new THREE.Scene();
    this.semanticScene = new THREE.Scene();
    [this.instanceScene, this.semanticScene].forEach((scene) => {
      scene.add(new THREE.HemisphereLight(0xffffff, 0xd6d9df, 2.2));
    });

    this.shared = new SharedCameraController();
    this.instanceControls = new TrackballControls(this.shared.camera, this.instanceRenderer.domElement);
    this.semanticControls = new TrackballControls(this.shared.camera, this.semanticRenderer.domElement);
    [this.instanceControls, this.semanticControls].forEach((controls) => {
      controls.rotateSpeed = 3.0;
      controls.zoomSpeed = 1.2;
      controls.panSpeed = 0.8;
      controls.staticMoving = false;
      controls.dynamicDampingFactor = 0.12;
    });
    this.instanceControls.target.copy(this.shared.target);
    this.semanticControls.target.copy(this.shared.target);

    this.loaded = false;
    this.disposed = false;
    this.loadedParts = 0;
    this.meshes = [];
    this.points = [];
    this.edges = [];
    this.animationId = null;
    this.syncing = false;
    this.showPoints = card.querySelector(".toggle-points").checked;
    this.showEdges = card.querySelector(".toggle-edges").checked;

    card.querySelector(".toggle-points").addEventListener("change", (event) => {
      this.showPoints = event.target.checked;
      this.updateOverlays();
    });
    card.querySelector(".toggle-edges").addEventListener("change", (event) => {
      this.showEdges = event.target.checked;
      this.updateOverlays();
    });

    this.instanceControls.addEventListener("change", () => {
      if (this.syncing) return;
      this.syncView("instance");
    });
    this.semanticControls.addEventListener("change", () => {
      if (this.syncing) return;
      this.syncView("semantic");
    });
  }

  syncView(source) {
    if (this.syncing) return;
    this.syncing = true;
    if (source === "instance") {
      this.copyCameraState(this.instanceControls, this.semanticControls);
    } else {
      this.copyCameraState(this.semanticControls, this.instanceControls);
    }
    this.syncing = false;
  }

  copyCameraState(fromControls, toControls) {
    this.shared.setTransform(fromControls.object, fromControls.target);
    const c = fromControls.object;
    toControls.object.position.copy(c.position);
    toControls.object.rotation.copy(c.rotation);
    toControls.object.up.copy(c.up);
    toControls.object.zoom = c.zoom;
    toControls.object.fov = c.fov;
    toControls.object.near = c.near;
    toControls.object.far = c.far;
    toControls.object.updateProjectionMatrix();
    toControls.target.copy(fromControls.target);
    toControls.update();
  }

  load() {
    if (this.loaded || this.disposed) return;
    this.loaded = true;

    if (this.instanceName) {
      this.loadOne(this.instanceName, this.instanceScene, this.instanceStatus, true);
    } else {
      this.instanceStatus.textContent = "缺少 GT 文件";
    }

    if (this.semanticName) {
      this.loadOne(this.semanticName, this.semanticScene, this.semanticStatus, false);
    } else {
      this.semanticStatus.textContent = "缺少语义文件";
    }
  }

  loadOne(name, scene, status, isInstance) {
    status.textContent = "加载中";
    loader.load(
      `${MODEL_BASE_URL}${safeName(name)}`,
      (geometry) => {
        if (this.disposed) {
          geometry.dispose();
          return;
        }

        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
          color: 0xdfe7f0,
          vertexColors: Boolean(geometry.getAttribute("color")),
          roughness: 0.7,
          metalness: 0.02,
          side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        const pointMaterial = new THREE.PointsMaterial({
          color: 0xffd400,
          size: this.overlayPointSize(geometry),
          sizeAttenuation: true,
          depthTest: true
        });
        const points = new THREE.Points(geometry, pointMaterial);
        scene.add(points);

        const edgeGeometry = new THREE.WireframeGeometry(geometry);
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: 0x000000,
          depthTest: true,
          transparent: true,
          opacity: 0.9
        });
        const edges = new THREE.LineSegments(edgeGeometry, edgeMaterial);
        scene.add(edges);

        this.meshes.push({ mesh, scene, geometry, isInstance });
        this.points.push({ points, mesh: points, isInstance });
        this.edges.push({ edges, mesh: edges, isInstance });
        this.loadedParts++;

        if (this.loadedParts === 1 && !this.frameReady) {
          this.frameReady = true;
          this.frameMesh(mesh, points, edges);
        }

        this.updateOverlays();
        status.textContent = this.summary(geometry);
        this.start();
      },
      (event) => {
        if (this.disposed) return;
        if (!event.total) return;
        status.textContent = `加载中 ${Math.round((event.loaded / event.total) * 100)}%`;
      },
      (error) => {
        if (this.disposed) return;
        console.error("PLY load failed", name, error);
        status.textContent = `加载失败 ${name}`;
      }
    );
  }

  summary(geometry) {
    const vertices = geometry.getAttribute("position")?.count ?? 0;
    const faces = geometry.index ? Math.floor(geometry.index.count / 3) : Math.floor(vertices / 3);
    return `${vertices.toLocaleString()} 顶点 / ${faces.toLocaleString()} 面`;
  }

  overlayPointSize(geometry) {
    geometry.computeBoundingBox();
    const size = geometry.boundingBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    return maxDim * 0.004;
  }

  frameMesh(mesh) {
    const box = new THREE.Box3().setFromObject(mesh);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const distance = maxDim / (2 * Math.tan(THREE.MathUtils.degToRad(this.shared.camera.fov) / 2));

    this.shared.target.copy(center);
    this.shared.camera.near = Math.max(maxDim / 1000, 0.001);
    this.shared.camera.far = maxDim * 1000;
    this.shared.camera.position.copy(center).add(new THREE.Vector3(distance * 0.72, distance * 0.42, distance));
    this.shared.camera.updateProjectionMatrix();

    this.instanceControls.target.copy(center);
    this.semanticControls.target.copy(center);
    this.instanceControls.update();
    this.semanticControls.update();
  }

  updateOverlays() {
    const pointVisible = this.showPoints;
    const edgeVisible = this.showEdges;
    [...this.points].forEach((entry) => {
      entry.points.visible = pointVisible;
    });
    [...this.edges].forEach((entry) => {
      entry.edges.visible = edgeVisible;
    });
  }

  resize() {
    [
      [this.instanceCanvas, this.instanceRenderer, this.instanceControls],
      [this.semanticCanvas, this.semanticRenderer, this.semanticControls]
    ].forEach(([canvas, renderer, controls]) => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width <= 0 || height <= 0) return;
      if (canvas.width !== width || canvas.height !== height) {
        renderer.setSize(width, height, false);
        controls.object.aspect = width / height;
        controls.object.updateProjectionMatrix();
        controls.handleResize();
      }
    });
  }

  start() {
    if (this.animationId || this.disposed) return;
    const animate = () => {
      if (this.disposed) return;
      this.resize();
      this.instanceControls.update();
      this.semanticControls.update();
      this.instanceRenderer.render(this.instanceScene, this.instanceControls.object);
      this.semanticRenderer.render(this.semanticScene, this.semanticControls.object);
      this.animationId = requestAnimationFrame(animate);
    };
    animate();
  }

  dispose() {
    this.disposed = true;
    if (this.animationId) cancelAnimationFrame(this.animationId);

    [
      [this.instanceControls, this.instanceRenderer, this.instanceCanvas],
      [this.semanticControls, this.semanticRenderer, this.semanticCanvas]
    ].forEach(([controls, renderer, canvas]) => {
      controls.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      if (canvas) {
        const fresh = canvas.cloneNode(false);
        canvas.replaceWith(fresh);
      }
    });

    this.meshes.forEach((entry) => {
      entry.scene.remove(entry.mesh);
      entry.geometry.dispose();
      entry.mesh.material.dispose();
    });
    this.points.forEach((entry) => {
      entry.points.geometry.dispose();
      entry.points.material.dispose();
      entry.mesh.remove(entry.points);
    });
    this.edges.forEach((entry) => {
      entry.edges.geometry.dispose();
      entry.edges.material.dispose();
      entry.mesh.remove(entry.edges);
    });

    this.meshes = [];
    this.points = [];
    this.edges = [];
    this.instanceStatus.textContent = "等待加载";
    this.semanticStatus.textContent = "等待加载";
  }
}

function basenameForModel(name) {
  if (!name.endsWith(INSTANCE_SUFFIX) && !name.endsWith(SEMANTIC_SUFFIX)) {
    return name;
  }
  if (name.endsWith(INSTANCE_SUFFIX)) {
    return name.slice(0, -INSTANCE_SUFFIX.length);
  }
  if (name.endsWith(SEMANTIC_SUFFIX)) {
    return name.slice(0, -SEMANTIC_SUFFIX.length);
  }
  return name;
}

function normalizeFavoriteName(name) {
  if (typeof name !== "string") return null;
  if (name.endsWith("_instance_colors_pred.ply")) {
    return name.slice(0, -"_instance_colors_pred.ply".length);
  }
  return basenameForModel(name);
}

function normalizeFavoriteList(items) {
  return new Set(
    items
      .map(normalizeFavoriteName)
      .filter(Boolean)
  );
}

function pairModels(files) {
  const map = new Map();
  files.forEach((file) => {
    if (!file.endsWith(".ply")) return;
    const key = basenameForModel(file);
    if (!map.has(key)) map.set(key, { instance: null, semantic: null, samples: [] });
    const item = map.get(key);
    item.samples.push(file);
    if (file.endsWith(INSTANCE_SUFFIX)) {
      item.instance = file;
    } else if (file.endsWith(SEMANTIC_SUFFIX)) {
      item.semantic = file;
    } else if (file.endsWith("_instance_colors_pred.ply")) {
      item.instance = file.replace("_instance_colors_pred.ply", INSTANCE_SUFFIX);
    }
  });

  return Array.from(map.entries())
    .map(([sample, payload]) => ({
      sample,
      instanceFile: payload.instance || "",
      semanticFile: payload.semantic || "",
      hasInstance: Boolean(payload.instance),
      hasSemantic: Boolean(payload.semantic)
    }))
    .sort((a, b) => a.sample.localeCompare(b.sample));
}

function formatPercent(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
}

function metricsMarkup(metrics, compact = false) {
  if (!metrics) return "";
  const macro = metrics.averaging === "macro" || metrics.macro_f1 !== undefined;
  const items = [
    ["Accuracy", metrics.accuracy],
    [macro ? "Macro Precision" : "Precision", macro ? metrics.macro_precision ?? metrics.precision : metrics.precision],
    [macro ? "Macro Recall" : "Recall", macro ? metrics.macro_recall ?? metrics.recall : metrics.recall],
    [macro ? "Macro F1" : "F1", macro ? metrics.macro_f1 ?? metrics.f1 : metrics.f1],
    ...(macro ? [["mIoU", metrics.macro_iou ?? metrics.iou]] : [])
  ];
  return `
    <div class="metric-grid${compact ? " compact" : ""}">
      ${items.map(([label, value]) => `
        <div class="metric-item">
          <span>${label}</span>
          <strong>${formatPercent(value, compact ? 2 : 4)}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function createCard(sampleName, instanceFile, semanticFile) {
  const card = document.createElement("article");
  card.className = "model-card sample-card";
  card.dataset.sample = sampleName;
  card.dataset.instance = instanceFile || "";
  card.dataset.semantic = semanticFile || "";

  const isFavorite = favorites.has(sampleName);
  const record = manifestSamples.get(sampleName);
  const displayName = record?.display_name ||
    (record?._datasetKey === "brepprediff-best-lxy"
      ? sampleName.replace(/^brepprediff_best_lxy__/, "")
      : sampleName);
  const gtTitle = record?._gtTitle || "SEG GT 高亮";
  const predictionTitle = record?._predictionTitle || "EBF/VBF 预测高亮";

  card.innerHTML = `
    <div class="model-title">
      <strong title="${escapeHtml(sampleName)}">${escapeHtml(displayName)}</strong>
      <div class="model-actions">
        <label class="overlay-toggle" title="显示黄色顶点">
          <input class="toggle-points" type="checkbox">
          <span>顶点</span>
        </label>
        <label class="overlay-toggle" title="显示黑色边线">
          <input class="toggle-edges" type="checkbox">
          <span>边线</span>
        </label>
        <button class="favorite-button${isFavorite ? " active" : ""}" type="button" title="星标收藏" aria-label="星标收藏 ${escapeHtml(sampleName)}" aria-pressed="${isFavorite}">
          ★
        </button>
        <button class="focus-button" type="button" title="放大聚焦" aria-label="放大聚焦 ${escapeHtml(sampleName)}">
          放大
        </button>
        <span>预测对比</span>
      </div>
    </div>
    ${record?.metrics || record?.binary_metrics ? `<div class="sample-metrics">${metricsMarkup(record.metrics || record.binary_metrics, true)}</div>` : ""}
    <div class="sample-grid" role="group" aria-label="实例与语义对比">
      <section class="viewer-panel" aria-label="${gtTitle}">
        <h3>${gtTitle}</h3>
        <div class="viewer-box">
          <canvas class="instance-view" aria-label="${escapeHtml(sampleName)} GT"></canvas>
          <div class="card-status sample-status instance-status">等待加载</div>
        </div>
      </section>
      <section class="viewer-panel" aria-label="${predictionTitle}">
        <h3>${predictionTitle}</h3>
        <div class="viewer-box">
          <canvas class="semantic-view" aria-label="${escapeHtml(sampleName)} 语义"></canvas>
          <div class="card-status sample-status semantic-status">等待加载</div>
        </div>
      </section>
    </div>
  `;

  return card;
}

function renderPrimitiveLegend(sampleName) {
  const legendItems = manifestSamples.get(sampleName)?._visualizationNumClasses === 2
    ? BINARY_LEGEND
    : PRIMITIVE_LEGEND;
  return legendItems.map((item) => `
    <div class="legend-row" title="${item.id} ${item.en} ${item.zh} RGB(${item.rgb})">
      <span class="legend-swatch" style="background: rgb(${item.rgb})"></span>
      <span class="legend-id">${item.id}</span>
      <span class="legend-name">${item.zh}</span>
    </div>
  `).join("");
}

function loadFavorites() {
  try {
    const raw = localStorage.getItem(FAVORITES_KEY);
    return normalizeFavoriteList(raw ? JSON.parse(raw) : []);
  } catch {
    return new Set();
  }
}

async function loadServerFavorites() {
  try {
    const response = await fetch("/api/favorites", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const serverFavorites = Array.isArray(data.favorites) ? data.favorites : [];

    if (serverFavorites.length === 0 && favorites.size > 0) {
      await saveFavorites();
    } else {
      favorites = normalizeFavoriteList(serverFavorites);
      localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites].sort()));
      renderCards();
    }
    serverFavoritesReady = true;
  } catch {
    serverFavoritesReady = false;
  }
}

async function saveFavorites() {
  localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites].sort()));
  try {
    const response = await fetch("/api/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ favorites: [...favorites].sort() })
    });
    serverFavoritesReady = response.ok;
  } catch {
    serverFavoritesReady = false;
  }
}

async function loadModelFiles() {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (response.ok) {
      const data = await response.json();
      if (Array.isArray(data.files) && data.files.length > 0) {
        activeFiles = data.files;
      }
    }
  } catch {
    // keep built-in MODEL_FILES if API fails
  } finally {
    await loadPredictionManifests();
    initAfterLoad();
  }
}

async function loadPredictionManifests() {
  const previousSignature = predictionManifestsSignature;
  const loaded = await Promise.all(PREDICTION_MANIFEST_SOURCES.map(async (source) => {
    try {
      const response = await fetch(
        `${MODEL_BASE_URL}${source.filename}?_=${Date.now()}`,
        { cache: "no-store" }
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return { source, manifest: await response.json() };
    } catch {
      return null;
    }
  }));

  predictionManifests = new Map();
  manifestSamples = new Map();
  loaded.filter(Boolean).forEach(({ source, manifest }) => {
    predictionManifests.set(source.key, { source, manifest });
    const samples = Array.isArray(manifest.samples) ? manifest.samples : [];
    samples.forEach((sample) => {
      manifestSamples.set(String(sample.sample_id), {
        ...sample,
        _datasetKey: source.key,
        _modelName: source.modelName,
        _visualizationNumClasses: source.visualizationNumClasses,
        _gtTitle: source.gtTitle,
        _predictionTitle: source.predictionTitle
      });
    });
  });
  predictionManifestsSignature = JSON.stringify(
    loaded.filter(Boolean).map(({ source, manifest }) => ({
      key: source.key,
      checkpoint: manifest.checkpoint,
      checkpointEpoch: manifest.checkpoint_epoch,
      metrics: manifest.metrics
    }))
  );
  renderMetricsSummary();
  return previousSignature !== null && previousSignature !== predictionManifestsSignature;
}

async function refreshPredictionManifests() {
  const changed = await loadPredictionManifests();
  if (changed) renderCards();
}

function renderMetricsSummary() {
  const selectedDataset = datasetFilter.value;
  const visibleKeys = selectedDataset === "all"
    ? PREDICTION_MANIFEST_SOURCES.map((source) => source.key)
    : [selectedDataset];
  const entries = visibleKeys
    .map((key) => predictionManifests.get(key))
    .filter((entry) => entry?.manifest?.metrics);
  if (!entries.length) {
    metricsSummary.hidden = true;
    metricsSummary.innerHTML = "";
    return;
  }
  metricsSummary.classList.toggle("multi-experiment", entries.length > 1);
  metricsSummary.innerHTML = entries.map(({ source, manifest }) => {
    const metrics = manifest.metrics;
    const evaluatedSamples = Number(manifest.evaluated_samples ?? 0).toLocaleString();
    const faces = Number(metrics.faces ?? 0).toLocaleString();
    const checkpoint = String(manifest.checkpoint ?? "unknown");
    const checkpointName = checkpoint.split("/").slice(-4, -2).join(" / ") || checkpoint;
    const checkpointEpoch = Number(manifest.checkpoint_epoch ?? 0);
    return `
      <section class="metrics-experiment ${source.key}">
        <div class="metrics-heading">
          <div>
            <strong>${escapeHtml(source.metricsTitle)}</strong>
            <span>${escapeHtml(source.metricsSubtitle || "正类：过渡面")} · ${escapeHtml(checkpointName)} · epoch ${checkpointEpoch}</span>
          </div>
          <span>${evaluatedSamples} 个样本 / ${faces} 个 OCC 面</span>
        </div>
        ${metricsMarkup(metrics)}
      </section>
    `;
  }).join("");
  metricsSummary.hidden = false;
}

function datasetForSample(sampleName) {
  const manifestDataset = manifestSamples.get(sampleName)?._datasetKey;
  if (manifestDataset) return manifestDataset;
  if (sampleName.startsWith("filletrec_model__")) {
    return "filletrec-model-filletrec-testset";
  }
  if (sampleName.startsWith("filletrec__")) return "filletrec";
  if (sampleName.startsWith("brepprediff_best_lxy__")) return "brepprediff-best-lxy";
  if (sampleName.startsWith("cjq_step_numeric__")) return "cjq-step-numeric";
  // Once the finetune-test manifest is available, omit stale duplicate PLYs
  // left by earlier path-deduplication runs (Ex9/Ex14 hash suffixes).
  return predictionManifests.has("finetune-test") ? "untracked" : "finetune-test";
}

function toggleFavorite(name, button) {
  if (favorites.has(name)) {
    favorites.delete(name);
  } else {
    favorites.add(name);
  }
  void saveFavorites();
  syncFavoriteButtons(name);
  if (favoritesOnly.checked) renderCards();
}

function syncFavoriteButtons(name) {
  document.querySelectorAll(".model-card").forEach((card) => {
    const favoriteId = card.dataset.sample;
    if (favoriteId !== name) return;
    const button = card.querySelector(".favorite-button");
    if (!button) return;
    const isFavorite = favorites.has(name);
    button.classList.toggle("active", isFavorite);
    button.setAttribute("aria-pressed", String(isFavorite));
  });
}

function attachSemanticLegend(card) {
  const semanticBox = card.querySelector(".semantic-view")?.closest(".viewer-box");
  if (!semanticBox || semanticBox.querySelector(".primitive-legend")) return;
  const legend = document.createElement("aside");
  legend.className = "primitive-legend";
  legend.setAttribute("aria-label", "预测类别颜色提示");
  legend.innerHTML = renderPrimitiveLegend(card.dataset.sample);
  semanticBox.appendChild(legend);
}

function openFocusViewer(name) {
  closeFocusViewer();

  const overlay = document.createElement("section");
  overlay.className = "focus-overlay";
  overlay.setAttribute("aria-label", "模型聚焦视图");
  overlay.innerHTML = `
    <div class="focus-panel">
      <button class="focus-close" type="button" title="关闭聚焦" aria-label="关闭聚焦">×</button>
    </div>
  `;

  const card = createCard(name, "", "");
  card.classList.add("focus-card");
  attachSemanticLegend(card);
  const focusButton = card.querySelector(".focus-button");
  focusButton.textContent = "返回";
  focusButton.title = "退出聚焦";
  focusButton.setAttribute("aria-label", `退出聚焦 ${name}`);
  overlay.querySelector(".focus-panel").appendChild(card);
  document.body.appendChild(overlay);
  document.body.classList.add("focus-open");

  // Find by exact dataset value so names with spaces, brackets, or CJK text keep working.
  const sourceCard = Array.from(document.querySelectorAll(".model-card"))
    .find((candidate) => candidate.dataset.sample === name && candidate !== card);
  if (sourceCard) {
    const instanceFile = sourceCard.dataset.instance;
    const semanticFile = sourceCard.dataset.semantic;
    const viewer = new PairCardViewer(card, instanceFile, semanticFile);
    viewers.set(card, viewer);
    viewer.load();

    overlay.querySelector(".focus-close").addEventListener("click", closeFocusViewer);
    overlay.querySelector(".favorite-button").addEventListener("click", (event) => toggleFavorite(name, event.currentTarget));
    overlay.querySelector(".focus-button").addEventListener("click", closeFocusViewer);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) closeFocusViewer();
    });
  }
}

function closeFocusViewer() {
  const overlay = document.querySelector(".focus-overlay");
  if (!overlay) return;
  const card = overlay.querySelector(".model-card");
  const viewer = viewers.get(card);
  if (viewer) {
    viewer.dispose();
    viewers.delete(card);
  }
  overlay.remove();
  document.body.classList.remove("focus-open");
}

function renderCards() {
  const keyword = searchInput.value.trim().toLowerCase();
  const selectedDataset = datasetFilter.value;
  const files = activeFiles;

  viewers.forEach((viewer) => viewer.dispose());
  viewers.clear();
  grid.innerHTML = "";

  if (!files.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = favoritesOnly.checked ? "还没有匹配的星标 PLY 文件" : "没有匹配的 PLY 文件";
    grid.appendChild(empty);
    return;
  }

  const pairs = pairModels(files);
  const matchedPairs = pairs.filter((p) => {
    const dataset = datasetForSample(p.sample);
    const datasetMatch = selectedDataset === "all" || selectedDataset === dataset;
    const keywordMatch = !keyword ||
      p.sample.toLowerCase().includes(keyword) ||
      p.instanceFile.toLowerCase().includes(keyword) ||
      p.semanticFile.toLowerCase().includes(keyword);
    const hasInstanceOrSemantic = p.hasInstance || p.hasSemantic;
    const isFavorite = favorites.has(p.sample);
    const favoriteMatch = !favoritesOnly.checked || isFavorite;
    return datasetMatch && keywordMatch && hasInstanceOrSemantic && favoriteMatch;
  });

  if (!matchedPairs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = favoritesOnly.checked ? "还没有匹配的星标样本" : "没有匹配的样本";
    grid.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  const groups = [
    {
      key: "finetune-test",
      title: "Finetune Test Split",
      description: "默认 BRepPreDiff baseline 在 finetune_test split 上的三分类结果",
      pairs: matchedPairs.filter((pair) => datasetForSample(pair.sample) === "finetune-test")
    },
    {
      key: "brepprediff-best-lxy",
      title: "BRepPreDiff Best · testset_lxy",
      description: "官方 test Macro-F1 0.968617 的扩充数据最优 MLP，在 testset_lxy 上的 SEG GT / 预测对比",
      pairs: matchedPairs.filter((pair) => datasetForSample(pair.sample) === "brepprediff-best-lxy")
    },
    {
      key: "cjq-step-numeric",
      title: "BRepPreDiff Best · cjq step_numeric",
      description: "3,993 个无 GT STEP：左侧原始模型，右侧为最优微调 MLP 的三分类预测",
      pairs: matchedPairs.filter((pair) => datasetForSample(pair.sample) === "cjq-step-numeric")
    },
    {
      key: "filletrec",
      title: "FilletRec Test Set",
      description: "VBF/EBF 合并为过渡面的二分类结果",
      pairs: matchedPairs.filter((pair) => datasetForSample(pair.sample) === "filletrec")
    },
    {
      key: "filletrec-model-filletrec-testset",
      title: "filletrec on filletrec testset",
      description: "FilletRec 在 FilletRec 官方 test split 上的二分类结果",
      pairs: matchedPairs.filter(
        (pair) => datasetForSample(pair.sample) === "filletrec-model-filletrec-testset"
      )
    },
    {
      key: "filletrec-ourtestset",
      title: "filletrec on ourtestset",
      description: "FilletRec 在 ourtestset 上的二分类过渡面预测结果",
      pairs: matchedPairs.filter((pair) => datasetForSample(pair.sample) === "filletrec-ourtestset")
    }
  ];

  groups.forEach((group) => {
    if (!group.pairs.length) return;
    const heading = document.createElement("header");
    heading.className = `dataset-heading ${group.key}`;
    heading.innerHTML = `
      <div>
        <h2>${group.title}</h2>
        <p>${group.description}</p>
      </div>
      <span>${group.pairs.length.toLocaleString()} 个样本</span>
    `;
    fragment.appendChild(heading);

    group.pairs.forEach((pair) => {
      const sampleName = pair.sample || pair.instanceFile || pair.semanticFile;
      const card = createCard(sampleName, pair.instanceFile, pair.semanticFile);
      attachSemanticLegend(card);
      fragment.appendChild(card);
    });
  });
  grid.appendChild(fragment);

  document.querySelectorAll(".favorite-button").forEach((button) => {
    const card = button.closest(".model-card");
    const name = card.dataset.sample;
    button.addEventListener("click", () => toggleFavorite(name, button));
  });

  document.querySelectorAll(".focus-button").forEach((button) => {
    const card = button.closest(".model-card");
    const name = card.dataset.sample;
    button.addEventListener("click", () => openFocusViewer(name));
  });

  observeCards();
}

function observeCards() {
  if (observer) observer.disconnect();
  observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const card = entry.target;
      const sampleName = card.dataset.sample;
      const instanceFile = card.dataset.instance;
      const semanticFile = card.dataset.semantic;

      if (entry.isIntersecting) {
        let viewer = viewers.get(card);
        if (!viewer) {
          viewer = new PairCardViewer(card, instanceFile, semanticFile);
          viewers.set(card, viewer);
        }
        viewer.load();
      } else {
        const viewer = viewers.get(card);
        if (viewer) {
          viewer.dispose();
          viewers.delete(card);
        }
      }
    });
  }, { rootMargin: "180px 0px" });

  document.querySelectorAll(".model-card").forEach((card) => observer.observe(card));
}

searchInput.addEventListener("input", renderCards);
datasetFilter.addEventListener("change", () => {
  renderMetricsSummary();
  renderCards();
});
favoritesOnly.addEventListener("change", renderCards);
window.addEventListener("resize", () => {
  viewers.forEach((viewer) => viewer.resize());
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeFocusViewer();
});

function initAfterLoad() {
  void loadServerFavorites();
  renderCards();
}

void loadModelFiles().then(() => {
  window.setInterval(() => void refreshPredictionManifests(), 10_000);
});
