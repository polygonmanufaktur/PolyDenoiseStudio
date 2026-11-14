# Changelog — PolyDenoise Studio

## [v1.5.1] — Viewport Performance, Filter Cleanup & Benchmark Polish

### ⚡ Viewport, Slider & Window-Resize Performance
* **GPU-Tonemapping im Viewport (`ModernGL`):** Standard-ACES-Filmic-Tonemapping inkl. Gamma 2.2 und IGN-Dithering läuft jetzt direkt als GLSL-Shader auf der GPU (~4 ms bei 2K statt >300 ms CPU), ergänzt durch einen gecachten **FP16-LUT** CPU-Fallback.
* **Flüssige Regler (`EV`, `Boost`, `Radius`):** Event-Coalescing (12–14 ms), Caching von Label-Updates & Leer-Pass-Checks sowie lazy Neuberechnung des Space/Klick-Vergleichsbilds sorgen für butterweiches Slider-Dragging.
* **Ruckelfreies Fenster-Resizing & Auto-Scrollbar:** Debounced `<Configure>`-Handling für die Preview-Leinwand, zustandsgeprüfte Canvas-Scrollbars und automatisches Ausblenden der linken Sidebar-Scrollbar, sobald alle Settings vertikal ins Fenster passen.

### ✂️ Filter- & UI-Bereinigung (17 fokussierte Kernels)
* **Filter-Konsolidierung:** 4 redundante Kernels (`MKS 2013 • Crisp HDR`, `Joint Cross-Bilateral • Gauss-Hermite`, `EWA Lanczos-3 Jinc Hybrid`, `Lanczos-4 • Unclamped`) inkl. Shader-, Fallback- und Hotkey-Code restlos entfernt. Die verbleibenden **17 Filter** sind lückenlos auf **`1`–`0` & `Ctrl+1`–`Ctrl+7`** nummeriert.
* **Toksvig Specular AA entfernt:** Checkbox sowie zugehöriger Normal-Varianz-/Roughness-Code mangels Roughness-Input-Pass entfernt.

### 📊 Benchmark & Zuverlässigkeit
* **Saubere Logs & lesbare Testnamen:** Compiler-URL-Links im Benchmark-Header entfernt und technische Test-IDs (z. B. `RT.hdr_calb_cnrm.1280x720`) in kompakte, im Log-Fenster nicht umbrechende Namen (`720p HDR • NoisyAux`) übersetzt.
* **Benchmark-Logik gefixt:** Monotone Fortschrittsanzeige über beide Geräte im `GPU + CPU`-Vergleichsmodus sowie sauberes Beenden laufender `oidnBenchmark.exe`-Prozesse beim Schließen des Fensters.

---

## [v1.5.0] — Modular Architecture, Perceptual Optics & Studio Inspectors

### 🧱 Modulare Codebasis & Tests
* **Paket-Split (`polydenoise/`):** Aufteilung des Monolithen in fokussierte Module (`constants`, `filter_registry`, `shaders_glsl`, `downscale_engine`, `oidn_backend`, `exr_pipeline`, `inspectors`, `ui_widgets`, `ui_app`) inkl. Launch-Kompatibilität über `start.bat` & `polydenoise_studio.py`.
* **Automatisierte Test-Suite (`tests/`):** Umfangreiche `unittest`-Abdeckung für Filter-Registry, GPU/CPU-Shader-Parität, Farbräume, ROI-Preview, Inspektoren sowie EXR- und Batch-Pipeline.

### 🔍 Filter-Suite, Inspektoren & Pipeline
* **Neue Optische & Perzeptuelle Filter:** `ICtCp HDR` (BT.2100 PQ), `Oklab`, `Guided Downscale (He et al. LDL^T)`, `AOV-DPID (Guide λ = 0)`, `EWA RobidouxSharp` & `EWA Mitchell` inkl. zweiter Ableitung für Z-Depth-Kantenschutz (`Depth Guide`).
* **ROI-Preview & Diagnose-Inspektoren:** Interaktiver `ROI Crop`-Testbereich (Rechtsklick-/Shift-Drag), `False-Color (EV)`, `Clipping & Fireflies`, `NaN / Inf / Neg / Albedo=0`-Overlay sowie Live-Cursor-Pixel-Probe.
* **EXR & Batch-Optimierungen:** Konsolidierter Multi-Layer EXR-Export (`<frame>_multilayer.exr`) und 1-Frame-Lookahead-Prefetch im Batch-Betrieb.

---

## [v1.4.0] — [Internal Non-Public Release] AOV-Prototypes & Stability
> Interner Meilenstein (nicht öffentlich), vollständig konsolidiert in `v1.5.0`.

* **Prototypen & UX:** Erste Experimente mit Albedo-Demodulation, `AOV-DPID`/`SAID` und `Guide λ`-Regler; scrollbare Sidebar mit fixiertem Footer, Statusleiste, `F`-Fit-Hotkey und Einstellungs-Persistenz (`oidn_config.json`).
* **Stabilität & Speicher:** Strikte `weakref`-basierte VRAM-Upload-Prüfung, begrenzter LRU-Guide-Cache, Thread-sichere Worker-UI-Queue (`SimpleQueue`) und VRAM-Freigabe nach Async-EXR-Writes.

---

## [v1.3.0] — [Internal Non-Public Release] GPU Shader Engine (`moderngl`)
> Interner Meilenstein (nicht öffentlich) zur Einführung der GLSL-GPU-Pipeline.

* **GPU-Beschleunigung:** Einführung der 2-Pass 32-Bit-GLSL-Pipeline (`ModernGL`) mit persistentem VRAM-Pool, separabler H/V-Konvolution und Thread-lokalen Batch-Kontexten.
* **Kernel- & Live-Preview-Grundlage:** Edge-MKS 2021, AMD CAS, Kaiser/Lanczos/Ginseng, `⚡ Preview Current Pass (RAM)`, `Boost`/`Radius`-Regler, `Hold M`-Maskenansicht sowie `Beauty Sum`.
