# Web character bundle

Extract all files together and open index.html, or serve the directory with any static server.
No API keys, backend, CDN, Cubism SDK, or network services are needed for animation.
Original artwork is represented by independent cropped sRGB RGBA textures.
Motion uses an editable procedural triangle-mesh rig. Blush/tears are drawn overlays.

Embedding on your site (serve model/assets from the same origin, or configure CORS):

```html
<canvas id="avatar" width="600" height="800"></canvas>
<script src="character/runtime.js"></script>
<script>
fetch('character/model.json').then(r => r.json()).then(async model => {
  const avatar = new Live2DWeb.Character(document.querySelector('#avatar'), model,
    {baseUrl: 'character/'});
  await avatar.load();
  const resize = () => {
    const bounds = avatar.canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1,
      8192 / bounds.width, 8192 / bounds.height);
    avatar.resize(bounds.width * ratio, bounds.height * ratio);
    if (!avatar.running) avatar.render(0);
  };
  const observer = window.ResizeObserver ? new ResizeObserver(resize) : null;
  observer?.observe(avatar.canvas);
  window.addEventListener('resize', resize); // Also supports browsers without ResizeObserver.
  resize();
  avatar.start();
  avatar.setExpression('smile');
  avatar.setParameters({mouthOpen: 0.5});
  window.avatar = avatar;
  // On component unmount: observer?.disconnect();
  // window.removeEventListener('resize', resize); avatar.destroy();
});
</script>
```

Character methods: load(), start(), stop(), destroy(), render(timestampMilliseconds),
setParameters(values), getParameters(), setExpression(name), setMotion(name), setAutoBlink(boolean),
setIdle(boolean), setPointerTracking(boolean), resize(width, height).
resize uses backing pixels; match CSS dimensions times devicePixelRatio to preserve
the character's proportions. Parameter ranges are in model.json.
Greeting waves an arm when assigned arm_right; otherwise it moves the head gently.
Names/IDs are metadata; filenames are hashes of IDs. Artwork rights remain with their owner.
