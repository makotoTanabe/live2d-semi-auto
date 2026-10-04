/* Local editor: credentials and session tokens exist only in this page's memory. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("editor-canvas");
  const ctx = canvas.getContext("2d");
  const roleLabels = {
    body: "胴体", face: "顔", hair_back: "後ろ髪", hair_front: "前髪", hair_side: "サイド髪",
    eye: "目", iris: "瞳", eyebrow: "眉", mouth: "口", accessory_head: "頭のアクセサリー",
    accessory_body: "体のアクセサリー", arm_left: "左腕", arm_right: "右腕", leg: "脚", static: "固定",
  };
  const state = {
    token: null, project: null, selected: null, busy: 0, dirty: false,
    images: {}, imageUrls: [], runtimeUrls: [], character: null, runtimeModel: null,
    renderTicket: 0, runtimeRevision: null, zoom: 1, pan: [0, 0], fitted: false, erase: false,
    pointer: null, stroke: [], job: null, pollTimer: null, playing: true,
  };

  function say(message) { $("status").textContent = message; }
  function error(message) {
    $("error-message").textContent = message || "処理に失敗しました。編集状態は保持されています。操作をやり直してください。";
    $("error-banner").hidden = false;
    say("操作を完了できませんでした");
  }
  function clearError() { $("error-banner").hidden = true; }
  function setDirty(dirty) {
    state.dirty = dirty;
    $("save-status").textContent = dirty ? "未保存の変更があります" : "原画を保持・画像は明示操作時のみ送信";
  }
  function hasProject() { return Boolean(state.project?.has_project); }
  function selectedPart() { return state.project?.parts?.find((part) => part.id === state.selected); }
  function locked() { return state.busy > 0 || Boolean(state.job); }
  function setEnabled() {
    document.querySelectorAll(".needs-project").forEach((element) => { element.disabled = !hasProject() || locked(); });
    document.querySelectorAll(".needs-part").forEach((element) => { element.disabled = !selectedPart() || locked(); });
    document.querySelectorAll(".visibility").forEach((element) => { element.disabled = locked(); });
    $("undo").disabled = locked() || !state.project?.can_undo;
    $("redo").disabled = locked() || !state.project?.can_redo;
    $("fit").disabled = !hasProject();
    for (const id of ["image-input", "project-input", "sample-alignment", "load-alignment", "sample-automatic", "sample-original"]) $(id).disabled = locked() || !state.token;
    $("job-accept").disabled = state.job?.status !== "ready";
    const lamaOption = $("repair-backend").querySelector('option[value="lama"]');
    lamaOption.disabled = !state.project?.capabilities?.lama_available;
    lamaOption.textContent = lamaOption.disabled ? "LaMa AI補完（モデル未設定）" : "LaMa AI補完（ローカル）";
  }
  async function request(path, { method = "GET", body, raw = false } = {}) {
    const headers = {};
    if (state.token) headers["X-Session-Token"] = state.token;
    if (body !== undefined && !(body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(body);
    }
    const response = await fetch(path, { method, headers, body, cache: "no-store", credentials: "same-origin" });
    if (!response.ok) {
      let detail = "";
      try {
        const data = await response.json();
        detail = typeof data.detail === "string" ? data.detail : typeof data.message === "string" ? data.message : "";
      } catch (_) { /* Do not expose HTML error pages or request data. */ }
      throw new Error(detail || (response.status === 401 ? "接続が切れました。ページを再読み込みしてください。" : `処理を完了できませんでした（HTTP ${response.status}）。編集状態は保持されています。`));
    }
    return raw ? response : response.json();
  }
  async function guarded(action) {
    clearError();
    try { return await action(); } catch (cause) { error(cause.message); return null; }
  }
  async function transaction(action) {
    state.busy += 1;
    setEnabled();
    try { return await action(); } finally { state.busy -= 1; setEnabled(); }
  }
  async function mutate(path, body, method = "POST") {
    return transaction(async () => {
      const result = await request(path, { method, body });
      await updateState(result.state || result, true);
      return result;
    });
  }
  function revokeUrls(urls) { urls.forEach((url) => URL.revokeObjectURL(url)); urls.length = 0; }
  async function fetchImage(path, urls) {
    const response = await request(path, { raw: true });
    const url = URL.createObjectURL(await response.blob());
    urls.push(url);
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  }

  function updatePartFields() {
    const part = selectedPart();
    $("part-name").value = part?.name || "";
    $("part-kind").value = part?.kind || "";
    $("adjust-scale").value = 1;
    $("adjust-angle").value = 0;
    $("adjust-x").value = 0;
    $("adjust-y").value = 0;
    $("role-select").value = roleForPart(part?.id);
    setEnabled();
  }
  function roleForPart(id) {
    if (!id) return "static";
    const model = state.runtimeModel;
    const binding = state.project?.runtime_config?.bindings?.[id] || model?.config?.bindings?.[id] || model?.bindings?.[id];
    return typeof binding === "string" ? binding : binding?.role || model?.layers?.find((part) => part.id === id)?.role || "static";
  }
  function renderParts() {
    const parts = state.project?.parts || [];
    $("part-list").replaceChildren();
    $("parts-count").textContent = hasProject() ? `${parts.length} パーツ · 手前から表示` : "画像を読み込んで開始";
    [...parts].reverse().forEach((part) => {
      const item = document.createElement("li");
      item.className = `part-item${part.id === state.selected ? " selected" : ""}${part.visible ? "" : " hidden-part"}`;
      item.dataset.partId = part.id;
      item.setAttribute("aria-selected", String(part.id === state.selected));
      item.tabIndex = 0;
      const visibility = document.createElement("button");
      visibility.className = "visibility";
      visibility.textContent = part.visible ? "◉" : "○";
      visibility.title = part.visible ? "非表示にする" : "表示する";
      visibility.setAttribute("aria-label", `${part.name}を${part.visible ? "非表示" : "表示"}`);
      visibility.disabled = locked();
      visibility.addEventListener("click", (event) => {
        event.stopPropagation();
        guarded(() => mutate(`/api/parts/${encodeURIComponent(part.id)}`, { visible: !part.visible }, "PATCH"));
      });
      const title = document.createElement("div");
      title.className = "part-title";
      const name = document.createElement("div");
      name.className = "part-name";
      name.textContent = part.name;
      const kind = document.createElement("small");
      kind.textContent = part.kind || "未分類";
      title.append(name, kind);
      item.append(visibility, title);
      const select = () => {
        if (locked()) return;
        state.selected = part.id;
        renderParts();
        updatePartFields();
        guarded(refreshView);
      };
      item.addEventListener("click", select);
      item.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
      $("part-list").append(item);
    });
    updatePartFields();
  }
  async function updateState(project, dirty = false) {
    state.project = project;
    const parts = project.parts || [];
    if (!parts.some((part) => part.id === state.selected)) state.selected = parts.at(-1)?.id || null;
    if (dirty) setDirty(true);
    $("empty-state").hidden = hasProject();
    const size = project.size || [0, 0];
    $("image-info").textContent = hasProject() ? `${project.source_name || "キャラクター"} · ${size[0]} × ${size[1]} px` : "画像未読み込み";
    const capabilities = project.capabilities || {};
    if (!$("gpt-model").value && capabilities.model) $("gpt-model").value = capabilities.model;
    $("gpt-status").textContent = capabilities.gpt_configured ? "環境のキーを利用できます。入力したキーは保存しません。" : "利用するキーを入力してください。入力したキーは保存しません。";
    renderParts();
    setEnabled();
    await refreshView();
  }

  function canvasSize() {
    const dpr = window.devicePixelRatio || 1;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width * dpr));
    const height = Math.max(1, Math.round(bounds.height * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  }
  function fitView() {
    canvasSize();
    const size = state.project?.size;
    if (!size) return;
    state.zoom = Math.min(canvas.width / size[0], canvas.height / size[1]) * 0.92;
    state.pan = [(canvas.width - size[0] * state.zoom) / 2, (canvas.height - size[1] * state.zoom) / 2];
    state.fitted = true;
    draw();
  }
  function tintMask(image) {
    const mask = document.createElement("canvas");
    mask.width = image.naturalWidth; mask.height = image.naturalHeight;
    const maskCtx = mask.getContext("2d");
    maskCtx.drawImage(image, 0, 0);
    const pixels = maskCtx.getImageData(0, 0, mask.width, mask.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const alpha = Math.round(pixels.data[i] * 0.45);
      pixels.data[i] = $("mask-target").value === "hidden" ? 55 : 42;
      pixels.data[i + 1] = $("mask-target").value === "hidden" ? 205 : 137;
      pixels.data[i + 2] = 255; pixels.data[i + 3] = alpha;
    }
    maskCtx.putImageData(pixels, 0, 0);
    return mask;
  }
  function draw() {
    if ($("view-mode").value === "live" && state.character) return;
    canvasSize();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!hasProject()) return;
    ctx.setTransform(state.zoom, 0, 0, state.zoom, state.pan[0], state.pan[1]);
    const ratio = canvas.width / canvas.getBoundingClientRect().width;
    canvas.dataset.scale = String(state.zoom / ratio);
    canvas.dataset.offsetX = String(state.pan[0] / ratio);
    canvas.dataset.offsetY = String(state.pan[1] / ratio);
    canvas.dataset.sourceWidth = String(state.project.size[0]);
    canvas.dataset.sourceHeight = String(state.project.size[1]);
    const mode = $("view-mode").value;
    if (mode === "overlay") {
      if (state.images.source) ctx.drawImage(state.images.source, 0, 0);
      if (state.images.maskTint) ctx.drawImage(state.images.maskTint, 0, 0);
    } else if (state.images.view) ctx.drawImage(state.images.view, 0, 0);
    if (state.stroke.length && state.pointer?.editing) {
      ctx.beginPath();
      state.stroke.forEach(([x, y], index) => { index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      if ($("mask-tool").value === "lasso") { ctx.closePath(); ctx.fillStyle = state.erase ? "#ff597744" : "#4d95ff44"; ctx.fill(); ctx.lineWidth = 2 / state.zoom; }
      else { ctx.lineWidth = Number($("brush-radius").value) * 2; ctx.lineCap = "round"; ctx.lineJoin = "round"; }
      ctx.strokeStyle = state.erase ? "#fa5f7999" : "#338fff99";
      ctx.stroke();
      if (state.stroke.length === 1 && $("mask-tool").value === "brush") {
        ctx.beginPath(); ctx.arc(...state.stroke[0], Number($("brush-radius").value), 0, Math.PI * 2);
        ctx.fillStyle = ctx.strokeStyle; ctx.fill();
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    $("zoom-label").textContent = `${Math.round(state.zoom / (window.devicePixelRatio || 1) * 100)}%`;
  }
  function destroyRuntime() {
    if (state.character) state.character.destroy();
    state.character = null;
    state.runtimeModel = null;
    state.runtimeRevision = null;
    revokeUrls(state.runtimeUrls);
  }
  function parameters() {
    return { angleX: Number($("angle-x").value), angleY: Number($("angle-y").value), angleZ: Number($("angle-z").value), eyeOpen: Number($("eye-open").value), mouthOpen: Number($("mouth-open").value), breath: Number($("breath").value) };
  }
  function applyRuntimeControls() {
    const character = state.character;
    if (!character) return;
    character.setParameters(parameters());
    character.setExpression($("expression").value);
    character.setMotion?.($("motion").value);
    character.setAutoBlink($("auto-blink").checked);
    character.setIdle($("idle").checked);
    character.setPointerTracking($("pointer-tracking").checked);
    if (state.playing && $("view-mode").value === "live") character.start();
    else { character.stop(); if ($("view-mode").value === "live") character.render(0); }
  }
  async function loadRuntime(ticket) {
    const model = await request("/api/runtime/model");
    const urls = [];
    const assetUrls = Object.create(null);
    try {
      const layers = model.layers || model.parts || [];
      await Promise.all(layers.map(async (layer) => {
        const response = await request(`/api/runtime/layers/${encodeURIComponent(layer.id)}.png`, { raw: true });
        const url = URL.createObjectURL(await response.blob());
        urls.push(url);
        assetUrls[layer.id] = url;
      }));
      if (ticket !== state.renderTicket) { revokeUrls(urls); return; }
      destroyRuntime();
      state.runtimeUrls = urls;
      state.runtimeModel = model;
      state.runtimeRevision = state.project.revision;
      canvasSize();
      const character = new window.Live2DWeb.Character(canvas, model, { assetUrls, autoBlink: $("auto-blink").checked, idle: $("idle").checked, pointerTracking: $("pointer-tracking").checked });
      state.character = character;
      await character.load();
      if (ticket !== state.renderTicket) { character.destroy(); return; }
      updatePartFields();
      applyRuntimeControls();
      $("zoom-label").textContent = "自動";
      say("表情とモーションを再生できます。役割を変更して動きを調整してください。");
    } catch (cause) { revokeUrls(urls); throw cause; }
  }
  async function refreshView() {
    const ticket = ++state.renderTicket;
    const mode = $("view-mode").value;
    $("canvas-wrap").classList.toggle("editing", mode === "overlay");
    $("stage-help").textContent = mode === "live" ? "表情・モーションを切り替えて確認" : mode === "overlay" ? "描画でマスクを編集 · ホイールで拡大 · 中ボタンで移動" : "ホイールで拡大 · 中ボタンで移動";
    if (!hasProject()) { destroyRuntime(); draw(); return; }
    $("canvas-loading").hidden = false;
    try {
      if (mode === "live") {
        if (state.character?.loaded && state.runtimeRevision === state.project.revision) applyRuntimeControls();
        else await loadRuntime(ticket);
      } else {
        destroyRuntime();
        const urls = [];
        const images = {};
        try {
          if (mode === "overlay") {
            images.source = await fetchImage("/api/image/source.png", urls);
            if (state.selected) {
              const target = $("mask-target").value === "hidden" ? "hidden" : "mask";
              images.mask = await fetchImage(`/api/image/${target}.png?part_id=${encodeURIComponent(state.selected)}`, urls);
              images.maskTint = tintMask(images.mask);
            }
          } else {
            const query = mode === "layer" && state.selected ? `?part_id=${encodeURIComponent(state.selected)}` : "";
            images.view = await fetchImage(`/api/image/${mode === "layer" && !state.selected ? "source" : mode}.png${query}`, urls);
          }
          if (ticket !== state.renderTicket) { revokeUrls(urls); return; }
          revokeUrls(state.imageUrls);
          state.imageUrls = urls;
          state.images = images;
          if (!state.fitted) fitView(); else draw();
        } catch (cause) { revokeUrls(urls); throw cause; }
      }
    } finally { if (ticket === state.renderTicket) $("canvas-loading").hidden = true; }
  }

  function confirmReplace() { return !state.dirty || window.confirm("未保存の変更があります。別の画像・プロジェクトを開くと、このページの編集内容は置き換わります。続けますか？"); }
  async function importFile(input, kind) {
    const file = input.files[0];
    if (!file || !confirmReplace()) { input.value = ""; return; }
    await transaction(async () => {
      say("読み込んでいます…");
      const body = new FormData(); body.set("file", file);
      const project = await request(`/api/import/${kind}`, { method: "POST", body });
      state.fitted = false;
      setDirty(false);
      $("view-mode").value = kind === "image" ? "source" : "live";
      await updateState(project.state || project);
      say(kind === "image" ? "画像を読み込みました。パーツ候補の作成、または手動編集を始められます。" : "プロジェクトを読み込みました。");
    });
    input.value = "";
  }
  async function sample(name) {
    if (!confirmReplace()) return;
    await transaction(async () => {
      say("サンプルを読み込んでいます…");
      const project = await request("/api/sample", { method: "POST", body: { name } });
      state.fitted = false;
      setDirty(false);
      $("view-mode").value = name === "original" ? "source" : "live";
      await updateState(project.state || project);
    });
  }
  function pointerPixels(event) {
    const bounds = canvas.getBoundingClientRect();
    return [(event.clientX - bounds.left) * canvas.width / bounds.width, (event.clientY - bounds.top) * canvas.height / bounds.height];
  }
  function sourcePoint(event) {
    const pixel = pointerPixels(event);
    const [width, height] = state.project.size;
    return [Math.round(Math.max(0, Math.min(width - 1, (pixel[0] - state.pan[0]) / state.zoom))), Math.round(Math.max(0, Math.min(height - 1, (pixel[1] - state.pan[1]) / state.zoom)))];
  }
  canvas.addEventListener("pointerdown", (event) => {
    if (!hasProject() || $("view-mode").value === "live" || locked()) return;
    const editing = event.button === 0 && $("view-mode").value === "overlay" && selectedPart();
    const panning = event.button === 1 || (event.button === 0 && event.altKey);
    if (!editing && !panning) return;
    event.preventDefault();
    canvas.focus();
    canvas.setPointerCapture(event.pointerId);
    state.pointer = { id: event.pointerId, editing: Boolean(editing && !panning), pixel: pointerPixels(event), pan: [...state.pan], partId: state.selected };
    state.stroke = state.pointer.editing ? [sourcePoint(event)] : [];
    $("canvas-wrap").classList.toggle("panning", panning);
    draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    if (state.pointer.editing) {
      const point = sourcePoint(event);
      const last = state.stroke.at(-1);
      if (state.stroke.length >= 4096) state.stroke = state.stroke.filter((_, index) => index % 2 === 0);
      if (Math.hypot(point[0] - last[0], point[1] - last[1]) >= 0.5) state.stroke.push(point);
    } else {
      const current = pointerPixels(event);
      state.pan = state.pointer.pan.map((value, index) => value + current[index] - state.pointer.pixel[index]);
    }
    draw();
  });
  async function finishStroke(event, cancelled = false) {
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    const pointer = state.pointer;
    const points = state.stroke;
    state.pointer = null;
    state.stroke = [];
    $("canvas-wrap").classList.remove("panning");
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    draw();
    if (cancelled || !pointer.editing || !points.length) return;
    if ($("mask-tool").value === "lasso" && points.length < 3) { say("投げ縄は3点以上の領域を囲んでください。"); return; }
    await guarded(() => mutate(`/api/parts/${encodeURIComponent(pointer.partId)}/mask`, { points, radius: Number($("brush-radius").value), erase: state.erase, hidden: $("mask-target").value === "hidden", tool: $("mask-tool").value }));
  }
  canvas.addEventListener("pointerup", (event) => finishStroke(event));
  canvas.addEventListener("pointercancel", (event) => finishStroke(event, true));
  canvas.addEventListener("wheel", (event) => {
    if (!hasProject() || $("view-mode").value === "live") return;
    event.preventDefault();
    const point = pointerPixels(event);
    const previous = state.zoom;
    state.zoom = Math.max(0.02, Math.min(30, state.zoom * Math.exp(-event.deltaY * 0.0015)));
    state.pan = point.map((value, index) => value - (value - state.pan[index]) * state.zoom / previous);
    draw();
  }, { passive: false });
  canvas.addEventListener("auxclick", (event) => { if (event.button === 1) event.preventDefault(); });

  async function history(direction) {
    if (locked() || !state.project?.[direction === "undo" ? "can_undo" : "can_redo"]) return;
    await mutate("/api/history", { direction });
  }
  async function exportFile(format) {
    await transaction(async () => {
      say("書き出しています…");
      const response = await request(`/api/export/${format}`, { method: "POST", body: {}, raw: true });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const disposition = response.headers.get("Content-Disposition") || "";
      const utfName = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
      const plainName = /filename="([^"]+)"/i.exec(disposition);
      let filename = format === "project" ? "character.l2split" : format === "psd" ? "character.psd" : `${format === "web" ? "web-character" : "parts"}.zip`;
      if (utfName) { try { filename = decodeURIComponent(utfName[1]); } catch (_) { /* safe default */ } }
      else if (plainName) filename = plainName[1];
      link.download = filename.replace(/[\\/\x00-\x1f]/g, "_");
      link.href = url;
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      if (format === "project") setDirty(false);
      say(format === "web" ? "Web用モデルを書き出しました。展開したデモで表情とモーションを再生できます。" : "書き出しが完了しました。");
    });
  }

  function remoteConsent(alignment = false) {
    const model = $("gpt-model").value.trim();
    if (!model) throw new Error("GPTモデル名を入力してください。");
    return window.confirm(`外部APIによる${alignment ? "位置合わせ" : "パーツ推定"}\n\n送信先: api.openai.com\n送信内容: ${alignment ? "現在の完成画像と、選択した分割パーツ配置図" : "現在の元画像"}\nモデル: ${model}\nAPIの利用料金が発生します。\n\n候補は自動採用されず、現在の編集内容は保持されます。送信しますか？`);
  }
  async function createJob(path, body) {
    await transaction(async () => {
      const job = await request(path, { method: "POST", body });
      state.job = job;
      $("proposal-title").textContent = job.kind === "alignment" ? "位置合わせ候補を確認" : job.kind === "repair" ? "補完候補を確認" : "パーツ候補を確認";
      $("job-message").textContent = "処理しています。現在の編集内容は保持されています。破棄すると結果は採用されません。";
      $("proposal-parts").replaceChildren();
      $("proposal-metadata").textContent = "";
      $("proposal-image").removeAttribute("src");
      $("job-accept").disabled = true;
      $("job-reject").textContent = "処理結果を破棄";
      $("proposal-dialog").showModal();
      const source = await fetchImage("/api/image/source.png", state.imageUrls);
      $("proposal-source").src = source.src;
      say("候補を作成しています…");
      await pollJob();
    });
  }
  async function pollJob() {
    const id = state.job?.id;
    if (!id) return;
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (state.job?.id !== id) return;
      state.job = job;
      $("job-message").textContent = job.message || (job.status === "ready" ? "候補ができました。確認して採用または破棄してください。" : "処理しています。編集内容は保持されています。");
      if (["gpt", "alignment"].includes(job.kind) && job.status === "running") $("job-message").textContent += " 候補を破棄しても、送信済みAPIの処理・料金は取り消せません。";
      if (job.status === "running" || job.status === "pending") {
        state.pollTimer = setTimeout(pollJob, 700);
      } else if (job.status === "ready") {
        const img = await fetchImage(`/api/jobs/${encodeURIComponent(id)}/preview.png`, state.imageUrls);
        if (state.job?.id !== id) return;
        $("proposal-image").src = img.src;
        $("proposal-parts").replaceChildren();
        (job.parts || []).forEach((part) => {
          const chip = document.createElement("span"); chip.className = "part-chip";
          chip.textContent = `${part.name || part.id}${part.confidence != null ? ` · 確信度 ${Math.round(part.confidence * 100)}%` : ""}`;
          $("proposal-parts").append(chip);
        });
        $("proposal-metadata").textContent = JSON.stringify(job.metadata || {}, null, 2);
        $("job-reject").textContent = "候補を破棄";
        say("候補を確認してください。採用するまで編集内容は変わりません。");
      } else {
        $("job-reject").textContent = "閉じる";
        if (job.status === "failed") error(job.message || "候補の作成に失敗しました。編集状態は保持されています。設定を確認して再実行してください。");
      }
      setEnabled();
    } catch (cause) {
      error(cause.message);
      $("job-message").textContent = "処理状況を取得できませんでした。候補を破棄して再実行してください。";
      $("job-reject").textContent = "候補を破棄";
    }
  }
  async function finishJob(accept) {
    const id = state.job?.id;
    if (!id) { $("proposal-dialog").close(); return; }
    clearTimeout(state.pollTimer);
    await transaction(async () => {
      const project = await request(`/api/jobs/${encodeURIComponent(id)}/${accept ? "accept" : "reject"}`, { method: "POST", body: {} });
      state.job = null;
      $("proposal-dialog").close();
      await updateState(project.state || project, accept);
      say(accept ? "候補を採用しました。各パーツの役割やマスクを修正できます。" : "候補を破棄しました。編集内容は保持されています。");
    });
  }

  function on(id, event, action) { $(id).addEventListener(event, (...args) => guarded(() => action(...args))); }
  on("dismiss-error", "click", clearError);
  on("image-input", "change", () => importFile($("image-input"), "image"));
  on("project-input", "change", () => importFile($("project-input"), "project"));
  on("sample-alignment", "click", () => sample("alignment"));
  on("load-alignment", "click", () => sample("alignment"));
  on("sample-automatic", "click", () => sample("automatic"));
  on("sample-original", "click", () => sample("original"));
  on("view-mode", "change", refreshView);
  on("fit", "click", () => { if ($("view-mode").value === "live") canvasSize(); else fitView(); });
  on("undo", "click", () => history("undo"));
  on("redo", "click", () => history("redo"));
  on("add-part", "click", async () => {
    await mutate("/api/parts", {});
    state.selected = state.project.parts.at(-1)?.id;
    $("view-mode").value = "overlay";
    $("mask-controls").open = true;
    renderParts(); await refreshView();
    $("part-name").focus();
  });
  on("apply-part", "click", async () => {
    if (!selectedPart()) return;
    const name = $("part-name").value.trim();
    const kind = $("part-kind").value.trim();
    if (!name) throw new Error("パーツ名を入力してください。");
    if (!kind) throw new Error("パーツの種類を入力してください。例: face、hair_front、body。");
    await mutate(`/api/parts/${encodeURIComponent(state.selected)}`, { name, kind }, "PATCH");
  });
  on("delete-part", "click", async () => {
    const part = selectedPart();
    if (part && window.confirm(`「${part.name}」を削除しますか？ 元画像は保持され、操作は取り消せます。`)) await mutate(`/api/parts/${encodeURIComponent(part.id)}`, {}, "DELETE");
  });
  on("move-up", "click", () => mutate(`/api/parts/${encodeURIComponent(state.selected)}/move`, { offset: 1 }));
  on("move-down", "click", () => mutate(`/api/parts/${encodeURIComponent(state.selected)}/move`, { offset: -1 }));
  on("role-select", "change", () => mutate("/api/runtime/config", { bindings: { ...state.project.runtime_config?.bindings, [state.selected]: { ...state.project.runtime_config?.bindings?.[state.selected], role: $("role-select").value } } }, "PATCH"));
  on("apply-adjustment", "click", () => createJob(`/api/parts/${encodeURIComponent(state.selected)}/adjustment`, { scale: Number($("adjust-scale").value), angle: Number($("adjust-angle").value), offset: [Number($("adjust-x").value), Number($("adjust-y").value)] }));
  on("mask-add", "click", () => {
    state.erase = false; $("mask-add").classList.add("active"); $("mask-erase").classList.remove("active");
    $("mask-add").setAttribute("aria-pressed", "true"); $("mask-erase").setAttribute("aria-pressed", "false");
  });
  on("mask-erase", "click", () => {
    state.erase = true; $("mask-erase").classList.add("active"); $("mask-add").classList.remove("active");
    $("mask-add").setAttribute("aria-pressed", "false"); $("mask-erase").setAttribute("aria-pressed", "true");
  });
  on("mask-target", "change", refreshView);
  on("brush-radius", "input", () => { $("brush-radius-value").textContent = `${$("brush-radius").value} px`; });
  on("repair-part", "click", () => createJob("/api/proposals/repair", { part_id: state.selected, backend: $("repair-backend").value }));
  on("auto-color", "click", () => createJob("/api/proposals/color", { count: Number($("color-count").value) }));
  on("propose-gpt", "click", async () => {
    if (!remoteConsent()) return;
    const body = { model: $("gpt-model").value.trim(), consent: true };
    if ($("api-key").value.trim()) body.api_key = $("api-key").value.trim();
    $("api-key").value = "";
    try { await createJob("/api/proposals/gpt", body); } finally { delete body.api_key; }
  });
  on("propose-alignment", "click", async () => {
    const file = $("atlas-input").files[0];
    if (!file) throw new Error("分割パーツの配置図を選択してください。");
    if (!remoteConsent(true)) return;
    const body = new FormData(); body.set("file", file); body.set("model", $("gpt-model").value.trim());
    body.set("background", $("atlas-background").value); body.set("consent", "true");
    if ($("api-key").value.trim()) body.set("api_key", $("api-key").value.trim());
    $("api-key").value = "";
    try { await createJob("/api/proposals/alignment", body); } finally { body.delete("api_key"); }
  });
  for (const format of ["project", "png", "psd", "web"]) on(`export-${format}`, "click", () => exportFile(format));
  on("job-accept", "click", () => finishJob(true));
  on("job-reject", "click", () => finishJob(false));
  on("job-reject-top", "click", () => finishJob(false));
  $("proposal-dialog").addEventListener("cancel", (event) => { event.preventDefault(); guarded(() => finishJob(false)); });
  const roleSelect = $("role-select");
  roleSelect.replaceChildren();
  Object.entries(roleLabels).forEach(([role, label]) => { const option = document.createElement("option"); option.value = role; option.textContent = label; roleSelect.append(option); });
  for (const id of ["expression", "motion", "idle", "auto-blink", "pointer-tracking"]) on(id, "change", applyRuntimeControls);
  for (const id of ["angle-x", "angle-y", "angle-z", "eye-open", "mouth-open", "breath"]) on(id, "input", () => { $(`${id}-value`).textContent = Number($(id).value).toFixed(2); applyRuntimeControls(); });
  on("playback", "click", () => {
    state.playing = !state.playing;
    $("playback").textContent = state.playing ? "一時停止" : "再生";
    $("playback").setAttribute("aria-pressed", String(state.playing));
    applyRuntimeControls();
  });
  on("reset-motion", "click", () => {
    for (const id of ["angle-x", "angle-y", "angle-z", "mouth-open"]) $(id).value = "0";
    $("eye-open").value = "1"; $("breath").value = "0.5"; $("expression").value = "neutral"; $("motion").value = "idle";
    for (const id of ["angle-x", "angle-y", "angle-z", "eye-open", "mouth-open", "breath"]) $(`${id}-value`).textContent = Number($(id).value).toFixed(2);
    applyRuntimeControls();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input,textarea,select") || $("proposal-dialog").open) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); guarded(() => history(event.shiftKey ? "redo" : "undo")); }
    else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") { event.preventDefault(); guarded(() => history("redo")); }
    else if (event.key.toLowerCase() === "f" && hasProject()) { event.preventDefault(); fitView(); }
  });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
  const resizeObserver = new ResizeObserver(() => {
    canvasSize();
    if ($("view-mode").value === "live") {
      state.character?.resize?.(canvas.width, canvas.height);
      if (!state.playing) state.character?.render(0);
    }
    else if (hasProject()) fitView();
  });
  resizeObserver.observe($("canvas-wrap"));
  setEnabled();
  guarded(() => transaction(async () => {
    const session = await request("/api/session", { method: "POST", body: {} });
    state.token = typeof session.session === "string" ? session.session : session.session?.token;
    if (!state.token) throw new Error("セッションを開始できませんでした。ページを再読み込みしてください。");
    await updateState(session.state || await request("/api/state"));
    say("準備できました。サンプルを動かすか、画像を読み込んでください。");
  }));
})();
