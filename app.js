'use strict';

/* ============================================================
 *  DOM
 * ============================================================ */
const $ = id => document.getElementById(id);
const canvas = $('core');
const spec = $('spectrum');
const sg = spec.getContext('2d');

/* ============================================================
 *  提前定义的一些玩意
 * ============================================================ */
let ctx, analyser, bins, wave, master, stream, source, buffer;
let started = 0, offset = 0, mode = 'demo', playing = false, loading = false;
let session = 0, rafTime = 0;
let angle = -0.32, tilt = 0.58, zoom = 1, drag = null;
let demoNodes = [];
let gain = 1.4, smoothing = 0.75, energy = 0;
let smoothPeakIndex = 0, smoothfundPeakIndex = 0, smooth = 0.8;
let seeking = false;
let fpsLast = 0, fpsFrames = 0;
let playStartCtxTime = 0;
let playStartOffset = 0;
let LOOP = true;
let vertOverflowWarned = false; // 顶点越界
let staticDirty = true;
let sortedRods = null;   // 缓存的深度排序结果
const MAX_STATIC_VERTS = 8192; // 极限2784
const MAX_DYN_VERTS    = 262144;   // 极限102096
const FFTSize = 2048;
const FBO_Bloom = 1;
const FBO_SS = 2;


/* ============================================================
 *  控制棒数据
 * ============================================================ */
const rods = [];
for (let x = -15; x <= 15; x++)
  for (let z = -15; z <= 15; z++) {
    const r = Math.hypot(x, z);
    if (r <= 15) rods.push({
      x, z, r,
      h: 0,
      seed: Math.abs(Math.sin(x * 127.1 + z * 311.7))
    });
  }
$('count').textContent = rods.length;
const N = rods.length;

/* ============================================================
 *  WebGL 初始化
 * ============================================================ */
const gl = canvas.getContext('webgl2', {
  antialias: true,
  alpha: true,
  premultipliedAlpha: true,
  powerPreference: 'high-performance'
});
if (!gl) {
  document.body.innerHTML = '<p style="color:#fff;padding:40px;font-family:sans-serif">当前浏览器不支持 WebGL2。</p>';
  throw new Error('no webgl2');
}

/* ---------- 主场景 Shader ---------- */
const VS = `#version 300 es
in vec3 aPos;
in vec3 aColor;
out vec3 vColor;
void main(){
  vColor = aColor;
  gl_Position = vec4(aPos, 1.0);
}`;

const FS = `#version 300 es
precision mediump float;
in vec3 vColor;
out vec4 fragColor;
void main(){
  fragColor = vec4(vColor, 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error('Shader compile error: ' + gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));

const A = {
  pos:   gl.getAttribLocation(prog, 'aPos'),
  color: gl.getAttribLocation(prog, 'aColor')
};


/* ============================================================
 *  Bloom 后处理模块（仅 SSAA）
 * ============================================================ */
const Bloom = (function () {

  let gl = null;
  let width = 1, height = 1;

  const MULTI_LEVEL = FBO_Bloom > 1;
  const LEVELS = MULTI_LEVEL ? Math.min(Math.max(FBO_Bloom, 2), 4) : 0;

  let bloomFBOs = [];
  let blurFBOs = [];
  let fboBright = null, fboBlur = null;
  let fboScene = null;

  let progBright = null, progBlur = null;
  let progCompositeMulti = null;
  let progCompositeSingle = null;
  let quadBuf = null;

  // ---- uniform location 缓存（修复项 6）----
  let U = null;

  let threshold = 0.25;
  let intensity = 1.8;
  let radius = 2.5;
  let singleRadius = 2.5;
  let sceneHasAlpha = true;

  /* ---------- Shader ---------- */

  const VS_QUAD = `#version 300 es
    in vec2 aPos;
    out vec2 vUV;
    void main(){
      vUV = aPos * 0.5 + 0.5;
      gl_Position = vec4(aPos, 0.0, 1.0);
    }`;

  const FS_BRIGHT = `#version 300 es
    precision mediump float;
    in vec2 vUV;
    uniform sampler2D uScene;
    uniform float uThreshold;
    out vec4 fragColor;
    void main(){
      vec4 s = texture(uScene, vUV);
      vec3 c = s.rgb;
      float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
      float d = max(lum, 0.0001);
      float f = max(lum - uThreshold, 0.0) / d;
      fragColor = vec4(c * f, 1.0);
    }`;

  const FS_BLUR = `#version 300 es
    precision mediump float;
    in vec2 vUV;
    uniform sampler2D uTexture;
    uniform vec2 uDirection;
    out vec4 fragColor;
    void main(){
      float w0 = 0.227027, w1 = 0.1945946, w2 = 0.1216216, w3 = 0.054054, w4 = 0.016216;
      vec3 result = texture(uTexture, vUV).rgb * w0;
      vec2 o1 = uDirection * 1.0;
      vec2 o2 = uDirection * 2.0;
      vec2 o3 = uDirection * 3.0;
      vec2 o4 = uDirection * 4.0;
      result += texture(uTexture, vUV + o1).rgb * w1;
      result += texture(uTexture, vUV - o1).rgb * w1;
      result += texture(uTexture, vUV + o2).rgb * w2;
      result += texture(uTexture, vUV - o2).rgb * w2;
      result += texture(uTexture, vUV + o3).rgb * w3;
      result += texture(uTexture, vUV - o3).rgb * w3;
      result += texture(uTexture, vUV + o4).rgb * w4;
      result += texture(uTexture, vUV - o4).rgb * w4;
      fragColor = vec4(result, 1.0);
    }`;

  const FS_COMPOSITE_MULTI = `#version 300 es
    precision mediump float;
    in vec2 vUV;
    uniform sampler2D uScene;
    uniform sampler2D uBloom0;
    uniform sampler2D uBloom1;
    uniform sampler2D uBloom2;
    uniform sampler2D uBloom3;
    uniform float uW0, uW1, uW2, uW3;
    uniform float uIntensity;
    out vec4 fragColor;
    void main(){
      vec4 scene = texture(uScene, vUV);
      vec3 b = texture(uBloom0, vUV).rgb * uW0
             + texture(uBloom1, vUV).rgb * uW1
             + texture(uBloom2, vUV).rgb * uW2
             + texture(uBloom3, vUV).rgb * uW3;
      b *= scene.a;
      vec3 color = scene.rgb + b * uIntensity;
      float maxC = max(max(color.r, color.g), color.b);
      color *= (1.0 - smoothstep(1.0, 1.6, maxC) * 0.4);
      fragColor = vec4(color, scene.a);
    }`;

  const FS_COMPOSITE_SINGLE = `#version 300 es
    precision mediump float;
    in vec2 vUV;
    uniform sampler2D uScene;
    uniform sampler2D uBloom;
    uniform float uIntensity;
    out vec4 fragColor;
    void main(){
      vec4 scene = texture(uScene, vUV);
      vec3 bloom = texture(uBloom, vUV).rgb;
      bloom *= scene.a;
      vec3 color = scene.rgb + bloom * uIntensity;
      float maxC = max(max(color.r, color.g), color.b);
      color *= (1.0 - smoothstep(1.0, 1.6, maxC) * 0.4);
      vec2 uv = vUV - 0.5;
      float vig = 1.0 - dot(uv, uv) * 1.6;
      color *= vig;
      fragColor = vec4(color, scene.a);
    }`;

  /* ---------- 工具函数 ---------- */

  function compile2(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error('Bloom shader error: ' + gl.getShaderInfoLog(s));
    return s;
  }

  function link(vsSrc, fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, compile2(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile2(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error('Bloom link error: ' + gl.getProgramInfoLog(p));
    return p;
  }

  function createFBO(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('FBO incomplete: 0x' + status.toString(16));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex, w, h };
  }

  function destroyFBO(f) {
    if (!f) return;
    gl.deleteFramebuffer(f.fbo);
    gl.deleteTexture(f.tex);
  }

  function bindTexture(tex, unit) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  function drawQuad(p) {
    const loc = gl.getAttribLocation(p, 'aPos');
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(loc);
  }

  function blurLevel(i, r) {
    const src = bloomFBOs[i];
    const tmp = blurFBOs[i];

    gl.bindFramebuffer(gl.FRAMEBUFFER, tmp.fbo);
    gl.viewport(0, 0, tmp.w, tmp.h);
    gl.useProgram(progBlur);
    bindTexture(src.tex, 0);
    gl.uniform1i(U.blur.uTexture, 0);
    gl.uniform2f(U.blur.uDirection, r / tmp.w, 0);
    drawQuad(progBlur);

    gl.bindFramebuffer(gl.FRAMEBUFFER, src.fbo);
    gl.viewport(0, 0, src.w, src.h);
    bindTexture(tmp.tex, 0);
    gl.uniform1i(U.blur.uTexture, 0);
    gl.uniform2f(U.blur.uDirection, 0, r / src.h);
    drawQuad(progBlur);
  }

  /* ---------- 公开 API ---------- */

  function init(glCtx, w, h, opts) {
    gl = glCtx;
    if (opts && opts.sceneHasAlpha !== undefined) sceneHasAlpha = opts.sceneHasAlpha;

    progBright = link(VS_QUAD, FS_BRIGHT);
    progBlur = link(VS_QUAD, FS_BLUR);
    progCompositeMulti = link(VS_QUAD, FS_COMPOSITE_MULTI);
    progCompositeSingle = link(VS_QUAD, FS_COMPOSITE_SINGLE);

    // ---- 一次性缓存所有 uniform location（修复项 6）----
    U = {
      bright: {
        uScene: gl.getUniformLocation(progBright, 'uScene'),
        uThreshold: gl.getUniformLocation(progBright, 'uThreshold')
      },
      blur: {
        uTexture: gl.getUniformLocation(progBlur, 'uTexture'),
        uDirection: gl.getUniformLocation(progBlur, 'uDirection')
      },
      compMulti: {
        uScene: gl.getUniformLocation(progCompositeMulti, 'uScene'),
        uIntensity: gl.getUniformLocation(progCompositeMulti, 'uIntensity'),
        uBloom: [0, 1, 2, 3].map(i => gl.getUniformLocation(progCompositeMulti, 'uBloom' + i)),
        uW:     [0, 1, 2, 3].map(i => gl.getUniformLocation(progCompositeMulti, 'uW' + i))
      },
      compSingle: {
        uScene: gl.getUniformLocation(progCompositeSingle, 'uScene'),
        uBloom: gl.getUniformLocation(progCompositeSingle, 'uBloom'),
        uIntensity: gl.getUniformLocation(progCompositeSingle, 'uIntensity')
      }
    };

    quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
      -1,  1,  1, -1,  1,  1,
    ]), gl.STATIC_DRAW);

    resize(w, h);
  }

  function resize(w, h) {
    width = w; height = h;

    destroyFBO(fboScene);
    destroyFBO(fboBright);
    destroyFBO(fboBlur);
    bloomFBOs.forEach(destroyFBO);
    blurFBOs.forEach(destroyFBO);
    bloomFBOs = [];
    blurFBOs = [];

    const SS = FBO_SS;
    const sw = Math.max(1, Math.round(width * SS));
    const sh = Math.max(1, Math.round(height * SS));
    fboScene = createFBO(sw, sh);

    if (MULTI_LEVEL) {
      for (let i = 0; i < LEVELS; i++) {
        const div = Math.pow(2, i + 1);
        const bw = Math.max(1, Math.floor(width / div));
        const bh = Math.max(1, Math.floor(height / div));
        bloomFBOs.push(createFBO(bw, bh));
        blurFBOs.push(createFBO(bw, bh));
      }
    } else {
      const bw = Math.max(1, Math.floor(width / 2));
      const bh = Math.max(1, Math.floor(height / 2));
      fboBright = createFBO(bw, bh);
      fboBlur = createFBO(bw, bh);
    }
  }

  function beginScene() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboScene.fbo);
    gl.viewport(0, 0, fboScene.w, fboScene.h);
    if (sceneHasAlpha) {
      gl.clearColor(0, 0, 0, 0);
    } else {
      gl.clearColor(0.071, 0.086, 0.078, 1);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  function endScene() {
    if (MULTI_LEVEL) endSceneMulti();
    else endSceneSingle();
  }

  function endSceneMulti() {
    const b0 = bloomFBOs[0];
    gl.bindFramebuffer(gl.FRAMEBUFFER, b0.fbo);
    gl.viewport(0, 0, b0.w, b0.h);
    gl.useProgram(progBright);
    bindTexture(fboScene.tex, 0);
    gl.uniform1i(U.bright.uScene, 0);
    gl.uniform1f(U.bright.uThreshold, threshold);
    drawQuad(progBright);

    for (let i = 0; i < LEVELS; i++) {
      if (i > 0) {
        const cur = bloomFBOs[i];
        const prev = bloomFBOs[i - 1];
        gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fbo);
        gl.viewport(0, 0, cur.w, cur.h);
        gl.useProgram(progBlur);
        bindTexture(prev.tex, 0);
        gl.uniform1i(U.blur.uTexture, 0);
        gl.uniform2f(U.blur.uDirection, 0, 0);
        drawQuad(progBlur);
      }
      blurLevel(i, radius);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(progCompositeMulti);

    bindTexture(fboScene.tex, 0);
    gl.uniform1i(U.compMulti.uScene, 0);

    const weights = [0.9, 0.5, 0.3, 0.15];
    for (let i = 0; i < 4; i++) {
      const tex = i < LEVELS ? bloomFBOs[i].tex : bloomFBOs[0].tex;
      bindTexture(tex, i + 1);
      gl.uniform1i(U.compMulti.uBloom[i], i + 1);
      gl.uniform1f(U.compMulti.uW[i], i < LEVELS ? weights[i] : 0);
    }
    gl.uniform1f(U.compMulti.uIntensity, intensity);
    drawQuad(progCompositeMulti);
  }

  function endSceneSingle() {
    const bw = fboBright.w, bh = fboBright.h;

    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBright.fbo);
    gl.viewport(0, 0, bw, bh);
    gl.useProgram(progBright);
    bindTexture(fboScene.tex, 0);
    gl.uniform1i(U.bright.uScene, 0);
    gl.uniform1f(U.bright.uThreshold, threshold);
    drawQuad(progBright);

    const BLUR_PASSES = 4;
    gl.useProgram(progBlur);
    for (let p = 0; p < BLUR_PASSES; p++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fboBlur.fbo);
      gl.viewport(0, 0, bw, bh);
      bindTexture(fboBright.tex, 0);
      gl.uniform1i(U.blur.uTexture, 0);
      gl.uniform2f(U.blur.uDirection, singleRadius / bw, 0);
      drawQuad(progBlur);

      gl.bindFramebuffer(gl.FRAMEBUFFER, fboBright.fbo);
      gl.viewport(0, 0, bw, bh);
      bindTexture(fboBlur.tex, 0);
      gl.uniform1i(U.blur.uTexture, 0);
      gl.uniform2f(U.blur.uDirection, 0, singleRadius / bh);
      drawQuad(progBlur);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(progCompositeSingle);
    bindTexture(fboScene.tex, 0);
    bindTexture(fboBright.tex, 1);
    gl.uniform1i(U.compSingle.uScene, 0);
    gl.uniform1i(U.compSingle.uBloom, 1);
    gl.uniform1f(U.compSingle.uIntensity, intensity);
    drawQuad(progCompositeSingle);
  }

  function setParams(p) {
    if (p.threshold !== undefined) threshold = p.threshold;
    if (p.intensity !== undefined) intensity = p.intensity;
    if (p.radius !== undefined) radius = p.radius;
  }

  return {
    init, resize, beginScene, endScene, setParams,
    get levels(){ return MULTI_LEVEL ? LEVELS : 1; },
    get ssaa()  { return FBO_SS; }
  };

})();

/* ============================================================
 *  顶点缓冲（修复项 3：静态 / 动态分离 + 越界保护）
 * ============================================================ */
const staticData = new Float32Array(MAX_STATIC_VERTS * 6);
const dynData    = new Float32Array(MAX_DYN_VERTS * 6);
let staticCount = 0;   // 顶点数（非浮点数）
let dynCount = 0;

const staticVBO = gl.createBuffer();
const dynVBO    = gl.createBuffer();

// 当前写入目标（由 push* 使用）
let writeBuf = null;     // Float32Array
let writeMax = 0;        // 最大浮点数
let vp = 0;              // 当前写入浮点索引

function setWriteTarget(buf, maxFloats) {
  writeBuf = buf;
  writeMax = maxFloats;
  vp = 0;
}

/* ============================================================
 *  等距投影（修复项：scale 缓存）
 * ============================================================ */
let canvasW = 1, canvasH = 1;
let projScale = 1;       // 每帧或相机变化时算一次
const camSin = { v: 0 }, camCos = { v: 1 };

function updateCameraCache() {
  projScale = Math.min(canvasW / 45, canvasH / 37) * zoom;
  camSin.v = Math.sin(angle);
  camCos.v = Math.cos(angle);
}

function project(x, z, h) {
  const rx = x * camCos.v - z * camSin.v;
  const rz = x * camSin.v + z * camCos.v;
  return [
    canvasW / 2 + rx * projScale,
    canvasH * 0.55 + rz * projScale * tilt - h * projScale
  ];
}

function toClip(px, py) {
  return [
    (px / canvasW) * 2 - 1,
    1 - (py / canvasH) * 2
  ];
}

/* ============================================================
 *  颜色工具（修复项 10：启用 hexCache）
 * ============================================================ */
const hexCache = new Map();
function rgb(r, g, b) {
  return [r / 255, g / 255, b / 255];
}
function hex2rgb(hex) {
  let v = hexCache.get(hex);
  if (v) return v;
  const n = parseInt(hex.slice(1), 16);
  v = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  hexCache.set(hex, v);
  return v;
}

const COL_HEX = {
  ringOuterDark: '#060b07',
  ringMain:      '#505948',
  ringInner:     '#878969',
  ringOuter:     '#303a2d',
  tickA:         '#484d31',
  tickB:         '#adac64',
  side1:         '#424838',
  side3:         '#333c2b',
  side4:         '#39432f',
  sideEdge:      '#111910',
  topEdge:       '#232d20',
  rodDot:        '#404735',
  edge:          '#0d1409'   // ← 修复项 1
};

const COL = Object.fromEntries(
  Object.entries(COL_HEX).map(([k, v]) => [k, hex2rgb(v)])
);

/* ============================================================
 *  顶点写入工具
 * ============================================================ */
function pushVertex(px, py, col) {
  // 越界保护 + 一次性警告
  if (vp + 6 > writeMax) {
    if (!vertOverflowWarned) {
      vertOverflowWarned = true;
      console.warn(
        'vertex buffer overflow:\n' +
        '  used: ' + (vp / 6) + ' vert\n' +
        '  max: ' + (writeMax / 6) + ' vert'
      );
    }
    return;
  }
  const cx = (px / canvasW) * 2 - 1;
  const cy = 1 - (py / canvasH) * 2;
  const b = writeBuf;
  b[vp++] = cx;
  b[vp++] = cy;
  b[vp++] = 0;
  b[vp++] = col[0];
  b[vp++] = col[1];
  b[vp++] = col[2];
}

function pushEdge(p1, p2, color, edgeWidth) {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const len = Math.hypot(dx, dy) || 1;
  const nX = -dy / len * edgeWidth * 0.5;
  const nY =  dx / len * edgeWidth * 0.5;

  const A0 = [p1[0] + nX, p1[1] + nY];
  const B0 = [p2[0] + nX, p2[1] + nY];
  const C0 = [p2[0] - nX, p2[1] - nY];
  const D0 = [p1[0] - nX, p1[1] - nY];

  pushVertex(A0[0], A0[1], color);
  pushVertex(B0[0], B0[1], color);
  pushVertex(C0[0], C0[1], color);
  pushVertex(A0[0], A0[1], color);
  pushVertex(C0[0], C0[1], color);
  pushVertex(D0[0], D0[1], color);
}

function pushQuad(p1, p2, p3, p4, col) {
  pushVertex(p1[0], p1[1], col);
  pushVertex(p2[0], p2[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p1[0], p1[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p4[0], p4[1], col);
}

function pushDisc(cx, cy, rx, ry, col, SEG) {
  SEG = SEG || 24;
  for (let i = 0; i < SEG; i++) {
    const a0 = i / SEG * Math.PI * 2;
    const a1 = (i + 1) / SEG * Math.PI * 2;
    pushVertex(cx, cy, col);
    pushVertex(cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, col);
    pushVertex(cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, col);
  }
}

/* ============================================================
 *  静态几何构建（修复项 3、4）
 *  外圈 + 刻度 + 每根棒的底座 quad 只在相机 / 视口变化时重建
 * ============================================================ */
function buildStaticGeometry() {
  updateCameraCache();
  setWriteTarget(staticData, staticData.length);

  // 1. 外圈环（修复项 4：像素宽度语义）
  function drawRing(radius, colorHex, widthPx) {
    const ringColor = hex2rgb(colorHex);
    const halfWidth = (widthPx * 0.5) / projScale;
    const step = 96;
    for (let i = 0; i < step; i++) {
      const a0 = i / step * Math.PI * 2;
      const a1 = (i + 1) / step * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const ri = radius - halfWidth;
      const ro = radius + halfWidth;
      const p0i = project(c0 * ri, s0 * ri, 0);
      const p1i = project(c1 * ri, s1 * ri, 0);
      const p0o = project(c0 * ro, s0 * ro, 0);
      const p1o = project(c1 * ro, s1 * ro, 0);
      pushQuad(p0i, p1i, p1o, p0o, ringColor);
    }
  }

  drawRing(16.4, '#060b07', 20);
  drawRing(16.5, '#505948',  2);
  drawRing(16.0, '#878969',  1);
  drawRing(17.3, '#303a2d',  1);

  // 2. 80 刻度块
  for (let i = 0; i < 80; i++) {
    const a = i / 80 * Math.PI * 2;
    const a2 = a + 0.034;
    const r0 = 15.7, r1 = 16.15;
    const p1 = project(Math.cos(a) * r0, Math.sin(a) * r0, 0);
    const p2 = project(Math.cos(a2) * r0, Math.sin(a2) * r0, 0);
    const p3 = project(Math.cos(a2) * r1, Math.sin(a2) * r1, 0);
    const p4 = project(Math.cos(a) * r1, Math.sin(a) * r1, 0);
    pushQuad(p1, p2, p3, p4, (i % 2 === 1) ? COL.tickA : COL.tickB);
  }

  // 3. 每根棒的底座顶面（h=0 时的 a0/b0/c0/d0）—— 静态
  const half = 0.44;
  for (let i = 0; i < N; i++) {
    const rod = rods[i];
    const x = rod.x, z = rod.z;
    const a0 = project(x - half, z - half, 0);
    const b0 = project(x + half, z - half, 0);
    const c0 = project(x + half, z + half, 0);
    const d0 = project(x - half, z + half, 0);
    rod._p0 = [a0, b0, c0, d0];   // 缓存，供动态部分复用
  }

  staticCount = vp / 6;

  gl.bindBuffer(gl.ARRAY_BUFFER, staticVBO);
  gl.bufferData(gl.ARRAY_BUFFER, staticData.subarray(0, vp), gl.STATIC_DRAW);
  
  // ← 新增：静态几何重建时同步更新深度排序（深度排序只随 angle 变）
  const sinA = camSin.v, cosA = camCos.v;
  sortedRods = [...rods].sort((a, b) =>
    (a.x * sinA + a.z * cosA) - (b.x * sinA + b.z * cosA)
  );
}

/* ============================================================
 *  动态几何构建（棒的顶面 / 侧面 / 边 / 圆点）
 * ============================================================ */
function buildDynamicGeometry() {
  setWriteTarget(dynData, dynData.length);
  updateCameraCache();

  // 深度排序（修复：按 angle 排序，只在相机变化时才需要重排；
  // 此处保留每帧排序以便未来支持棒高度影响顺序时扩展，但用缓存 sin/cos）
  const sinA = camSin.v, cosA = camCos.v;
  const sorted = sortedRods;

  const half = 0.44;
  const dotSize = projScale * 0.12;
  const edgeWidth = Math.max(0.8 * zoom, 1);
  const sa = sinA, ca = cosA;


  for (let s = 0; s < sorted.length; s++) {
    const rod = sorted[s];
    const amp = rod._amp || 0;
    const h = rod.h;
    const x = rod.x, z = rod.z;

    const a = project(x - half, z - half, h);
    const b = project(x + half, z - half, h);
    const c = project(x + half, z + half, h);
    const d = project(x - half, z + half, h);
    const [a0, b0, c0, d0] = rod._p0;

    // 4 个侧面
    pushQuad(a, b, b0, a0, COL.side1);
    pushQuad(b, c, c0, b0, rgb(55 + amp * 30, 62 + amp * 30, 44 + amp * 20));
    pushQuad(c, d, d0, c0, COL.side3);
    pushQuad(d, a, a0, d0, COL.side4);

    // 改为直接用 seed 重算一次（更稳）
    const seed = rod.seed;
    const top = seed > 0.92 ? [143, 109, 81]
              : seed > 0.85 ? [97, 118, 104]
              : [139 + seed * 29, 148 + seed * 22, 119 + seed * 21];
    pushQuad(a, b, c, d, rgb(
      top[0] + amp * 43,
      top[1] + amp * 40,
      top[2] + amp * 24
    ));

    // 顶面 4 边
    pushEdge(a, b, COL.edge, edgeWidth);
    pushEdge(b, c, COL.edge, edgeWidth);
    pushEdge(c, d, COL.edge, edgeWidth);
    pushEdge(d, a, COL.edge, edgeWidth);

    // 朝镜头的竖棱
    const depthA = -sa - ca, depthB = sa - ca, depthC = sa + ca, depthD = -sa + ca;
    const faces = [
      { front: (depthA + depthB) > (depthC + depthD), edge1: [a, a0], edge2: [b, b0] },
      { front: (depthB + depthC) > (depthD + depthA), edge1: [b, b0], edge2: [c, c0] },
      { front: (depthC + depthD) > (depthA + depthB), edge1: [c, c0], edge2: [d, d0] },
      { front: (depthD + depthA) > (depthB + depthC), edge1: [d, d0], edge2: [a, a0] },
    ];
    const seen = new Set();
    for (const f of faces) {
      if (!f.front) continue;
      for (const e of [f.edge1, f.edge2]) {
        const key = e[0] === a ? 'a' : e[0] === b ? 'b' : e[0] === c ? 'c' : 'd';
        if (seen.has(key)) continue;
        seen.add(key);
        pushEdge(e[0], e[1], COL.edge, edgeWidth);
      }
    }

    // 圆点
    const dotPos = project(x, z, h + 0.01);
    pushDisc(dotPos[0], dotPos[1], dotSize, dotSize * tilt, COL.rodDot, 32);
  }

  dynCount = vp / 6;

  gl.bindBuffer(gl.ARRAY_BUFFER, dynVBO);
  gl.bufferData(gl.ARRAY_BUFFER, dynData.subarray(0, vp), gl.DYNAMIC_DRAW);
}

/* ============================================================
 *  尺寸
 * ============================================================ */

function size() {
  const rect = canvas.getBoundingClientRect();
  const d = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * d));
  canvas.height = Math.max(1, Math.round(rect.height * d));
  canvasW = canvas.clientWidth;
  canvasH = canvas.clientHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);

  Bloom.resize(canvas.width, canvas.height);

  const srect = spec.getBoundingClientRect();
  spec.width = Math.max(1, Math.round(srect.width * d));
  spec.height = Math.max(1, Math.round(srect.height * d));
  sg.setTransform(d, 0, 0, d, 0, 0);

  staticDirty = true;   // 视口变化 → 静态几何重建
}

canvasW = canvas.clientWidth || 800;
canvasH = canvas.clientHeight || 600;
Bloom.init(gl, Math.max(1, canvas.width || 800), Math.max(1, canvas.height || 600), { sceneHasAlpha: true });

new ResizeObserver(size).observe(canvas);
new ResizeObserver(size).observe(spec);

/* ============================================================
 *  主循环
 * ============================================================ */
function frame(t) {
  const dt = Math.min((t - rafTime) / 16.67, 3) || 1;
  rafTime = t;
  if (!fpsLast) fpsLast = t;
  fpsFrames++;
  if (t - fpsLast >= 500) {
    $('fps').textContent = Math.round(fpsFrames * 1000 / (t - fpsLast));
    fpsFrames = 0;
    fpsLast = t;
  }

  // ---- 音频分析 ----
  if (analyser && playing) analyser.getByteFrequencyData(bins);
  let peak = 0, peakIndex = 0;
  let fundPeak = 0, fundIndex = 0;

  if (playing && bins) {
    for (let i = 2; i < bins.length - 2; i++) {
      if (bins[i] > fundPeak) { fundPeak = bins[i]; fundIndex = i; }
    }
    smoothPeakIndex = smoothPeakIndex * smooth + fundIndex * (1.0 - smooth);

    const sampleRate = ctx ? ctx.sampleRate : 44100;
    const binHz = sampleRate / FFTSize;
    const startHigh = Math.floor(300 / binHz);
    for (let i = startHigh; i < bins.length - 2; i++) {
      if (bins[i] > peak) { peak = bins[i]; peakIndex = i; }
    }
    smoothfundPeakIndex = smoothfundPeakIndex * smooth + peakIndex * (1.0 - smooth);
  }

  energy = 0;
  if (playing && wave) {
    analyser.getFloatTimeDomainData(wave);
    for (let i = 0; i < wave.length; i++) energy += wave[i] * wave[i];
    energy = Math.sqrt(energy / wave.length);
  }

  // ---- Bloom 动态参数 ----
  const bloomIntensity = Math.min(3.00, 1.35 + energy * 0.75);
  const bloomThreshold = Math.max(0.15, 0.40 - energy * 0.25);
  Bloom.setParams({
    intensity: bloomIntensity,
    threshold: bloomThreshold,
    radius: 1.5
  });

  // ---- 更新柱子高度 ----
  const sampleRate = ctx ? ctx.sampleRate : 44100;
  const binHz = sampleRate / FFTSize;

  for (let i = 0; i < N; i++) {
    const rod = rods[i];
    let amp = 0;
    if (playing && bins) {
      const hz = 45 * Math.pow(320, rod.r / 15);
      const idx = Math.max(1, Math.min(bins.length - 2, Math.round(hz / binHz)));
      amp = (bins[idx] + bins[idx - 1] + bins[idx + 1]) / 765;
      amp = Math.pow(amp, 1.7) * gain;
    } else if (!playing && mode === 'demo' && !ctx) {
      amp = 0.10 + 0.20 * (0.5 + 0.5 * Math.sin(rod.r * 0.6 - t * 0.0007 + rod.x * 0.1));
    }
    rod._amp = amp;
    const target = 0.07 + amp * 4.8;
    rod.h += (target - rod.h) * (1 - Math.pow(0.45 + smoothing * 0.53, dt));
  }

  // ---- 几何重建 ----
  if (staticDirty) {
    buildStaticGeometry();
    staticDirty = false;
  }
  buildDynamicGeometry();

  /* ==========================================================
   *  渲染：Bloom 包裹
   * ========================================================== */
  gl.disable(gl.DEPTH_TEST);
  Bloom.beginScene();

  gl.useProgram(prog);

  // 静态部分
  if (staticCount > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, staticVBO);
    gl.enableVertexAttribArray(A.pos);
    gl.vertexAttribPointer(A.pos, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(A.color);
    gl.vertexAttribPointer(A.color, 3, gl.FLOAT, false, 24, 12);
    gl.drawArrays(gl.TRIANGLES, 0, staticCount);
  }

  // 动态部分
  if (dynCount > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, dynVBO);
    gl.enableVertexAttribArray(A.pos);
    gl.vertexAttribPointer(A.pos, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(A.color);
    gl.vertexAttribPointer(A.color, 3, gl.FLOAT, false, 24, 12);
    gl.drawArrays(gl.TRIANGLES, 0, dynCount);
  }

  Bloom.endScene();

  /* ==========================================================
   *  频谱小图
   * ========================================================== */
  const sw = spec.clientWidth, sh = spec.clientHeight;
  sg.clearRect(0, 0, sw, sh);
  for (let i = 0; i < 48; i++) {
    const v = playing && bins
      ? bins[Math.min(bins.length - 1, Math.floor(Math.pow(i / 47, 2) * 700))] / 255
      : 0;
    sg.fillStyle = i < 12 ? '#d8b975' : i < 30 ? '#c3cf87' : '#95b1a1';
    sg.fillRect(i * sw / 48, sh - Math.max(2, v * sh), sw / 48 - 3, Math.max(2, v * sh));
  }

  // ---- 读数 ----
  if (t % 100 < 25) {
    $('level').innerHTML = (playing ? Math.round(20 * Math.log10(Math.max(energy, 0.001))) : '—') + '<small>dB</small>';
    $('fund').innerHTML = (playing && fundPeak > 0) ? Math.round(smoothPeakIndex * binHz) + '<small>Hz</small>' : '—<small>Hz</small>';
    $('freq').innerHTML = (playing && peak > 0) ? Math.round(smoothfundPeakIndex * binHz) + '<small>Hz</small>' : '—<small>Hz</small>';

    if (playing && !seeking) {
      let seconds;
      if (buffer && mode === 'file') {
        const elapsed = ctx.currentTime - playStartCtxTime;
        seconds = (playStartOffset + elapsed) % buffer.duration;
        $('seek').value = seconds / buffer.duration * 100;
      } else {
        seconds = offset + (ctx.currentTime - started);
      }
      $('elapsed').textContent = format(seconds);
    }
  }
  requestAnimationFrame(frame);
}

/* ============================================================
 *  音频系统
 * ============================================================ */
function format(s) {
  return Math.floor(s / 60).toString().padStart(2, '0') + ':' +
         Math.floor(s % 60).toString().padStart(2, '0');
}

async function audio() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = ctx.createAnalyser();
    analyser.fftSize = FFTSize;
    analyser.smoothingTimeConstant = 0.75;
    bins = new Uint8Array(analyser.frequencyBinCount);
    wave = new Float32Array(analyser.fftSize);
    master = ctx.createGain();
    master.gain.value = 0.4;
    analyser.connect(master);
    master.connect(ctx.destination);
  }
  await ctx.resume();
  // 空闲时释放音频设备（修复项 7，与 V2 一致）
  if (!playing && !loading && ctx.state === 'running') await ctx.suspend();
}

function stop() {
  session++;
  loading = false;
  if (source) {
    source.onended = null;
    try { source.stop?.(); } catch (e) {}
    source.disconnect();
    source = null;
  }
  demoNodes.forEach(n => { try { n.stop?.(); n.disconnect(); } catch (e) {} });
  demoNodes = [];
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  playing = false;
  if (ctx?.state === 'running') ctx.suspend().catch(() => {});   // 修复项 7
  update();
}

function update() {
  $('play').textContent = playing ? 'Ⅱ' : '▶';
  $('play').setAttribute('aria-label', playing ? '暂停' : '播放');
  $('demo').classList.toggle('selected', mode === 'demo');
  $('mic').classList.toggle('selected', mode === 'mic');
  $('status').textContent = playing
    ? '运行中 · ' + (mode === 'demo' ? '演示信号' : mode === 'mic' ? '实时输入' : '音频分析')
    : '待机 · ' + (mode === 'demo' ? '演示预览' : '已暂停');
  $('tracksub').textContent = playing ? '控制棒随频谱实时响应' : '点击播放，启动声场';
  $('seek').disabled = mode !== 'file';
}

async function start() {
  if (loading || playing) return;
  loading = true;
  const attempt = ++session;
  try {
    await audio();
    if (attempt !== session) return;

    if (mode === 'mic') {
      $('message').textContent = '等待麦克风授权…';
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (attempt !== session) { s.getTracks().forEach(t => t.stop()); return; }
      stream = s;
      source = ctx.createMediaStreamSource(s);
      source.connect(analyser);
      master.gain.value = 0;
      $('message').textContent = '正在监听麦克风；停止或切换输入即可关闭。';
    } else {
      master.gain.value = +$('volume').value / 100;

      if (mode === 'file' && buffer) {
        source = ctx.createBufferSource();
        source.buffer = buffer;
        source.loop = LOOP;
        source.connect(analyser);
        source.onended = () => {
          if (LOOP) return;
          stop();
          offset = 0;
          $('seek').value = 0;
          $('elapsed').textContent = '00:00';
        };
        source.start(0, offset);
        playStartCtxTime = ctx.currentTime;
        playStartOffset = offset;
      } else {
        [55, 82.41, 110, 164.81, 220, 329.63, 440, 659.25].forEach((freq, i) => {
          const osc = ctx.createOscillator();
          const v = ctx.createGain();
          const lfo = ctx.createOscillator();
          const lg = ctx.createGain();
          osc.type = i < 3 ? 'triangle' : 'sine';
          osc.frequency.value = freq;
          v.gain.value = 0.025;
          lfo.frequency.value = 0.22 + i * 0.087;
          lg.gain.value = 0.023;
          lfo.connect(lg);
          lg.connect(v.gain);
          osc.connect(v);
          v.connect(analyser);
          osc.start();
          lfo.start();
          demoNodes.push(osc, lfo, v, lg);
        });
      }
      $('message').textContent = '音频仅在本机处理，不会上传。';
    }

    started = ctx.currentTime;
    playing = true;
    update();
  } catch (e) {
    if (attempt !== session) return;
    stop();
    $('message').textContent = mode === 'mic'
      ? '无法访问麦克风，请检查浏览器权限，或导入音频。'
      : '无法播放此音频，请尝试其他文件。';
  } finally {
    if (attempt === session) loading = false;
  }
}

/* ============================================================
 *  事件绑定
 * ============================================================ */
$('play').onclick = () => {
  if (playing) {
    if (buffer && mode === 'file') {
      const elapsed = ctx.currentTime - playStartCtxTime;
      offset = (playStartOffset + elapsed) % buffer.duration;
    } else {
      offset += ctx.currentTime - started;
    }
    stop();
  } else start();
};

$('demo').onclick = () => {
  stop();
  mode = 'demo';
  offset = 0;
  buffer = null;
  $('trackname').textContent = 'REACTOR / 内置信号';
  $('duration').textContent = 'DEMO / LOOP';
  $('elapsed').textContent = '00:00';
  $('seek').value = 0;
  update();
  start();
};

$('mic').onclick = () => {
  stop();
  mode = 'mic';
  offset = 0;
  buffer = null;
  $('trackname').textContent = 'MIC / 麦克风输入';
  $('duration').textContent = 'LIVE';
  $('elapsed').textContent = '00:00';
  update();
  start();
};

$('upload').onclick = () => $('file').click();

$('file').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  stop();
  const id = session;
  loading = true;
  $('message').textContent = '正在读取音频…';
  try {
    await audio();
    if (id !== session) return;
    const decoded = await ctx.decodeAudioData(await f.arrayBuffer());
    if (id !== session) return;
    buffer = decoded;
    mode = 'file';
    offset = 0;
    $('trackname').textContent = f.name;
    $('duration').textContent = format(buffer.duration);
    $('elapsed').textContent = '00:00';
    $('seek').value = 0;
    update();
    loading = false;
    await start();
  } catch (err) {
    if (id !== session) return;
    stop();
    $('message').textContent = '无法读取该文件，请尝试 MP3 或 WAV 音频。';
  } finally {
    if (id === session) loading = false;
  }
};

document.addEventListener('keydown', e => {
  if (e.repeat) return;
  const el = document.activeElement;
  const tag = el?.tagName;
  const isTextInput =
    el && el !== document.body && el !== document.documentElement &&
    (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable);
  if (isTextInput) return;

  if (e.code === 'Space') {
    e.preventDefault();
    e.stopPropagation();
    $('play').click();
    return;
  }

  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  if (mode !== 'file' || !buffer) return;
  e.preventDefault();

  const STEP = 5;
  const dir = e.key === 'ArrowRight' ? 1 : -1;
  const cur = +$('seek').value / 100 * buffer.duration;
  const next = Math.max(0, Math.min(buffer.duration, cur + dir * STEP));
  $('seek').value = next / buffer.duration * 100;
  $('seek').dispatchEvent(new Event('input'));
  $('seek').dispatchEvent(new Event('change'));
});

$('seek').onkeydown = e => {
  if (mode !== 'file' || !buffer) {seeking = false;return};
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const STEP = 5;
  const dir = e.key === 'ArrowRight' ? 1 : -1;
  const cur = +$('seek').value / 100 * buffer.duration;
  const next = Math.max(0, Math.min(buffer.duration, cur + dir * STEP));
  $('seek').value = next / buffer.duration * 100;
  $('seek').dispatchEvent(new Event('input'));
  $('seek').dispatchEvent(new Event('change'));
};

$('seek').oninput = () => {
  if (mode !== 'file' || !buffer) {seeking = false;return};
  seeking = true;
  $('elapsed').textContent = format(+$('seek').value / 100 * buffer.duration);
};

$('seek').onchange = () => {
  if (mode !== 'file' || !buffer) return;
  const was = playing;
  stop();
  offset = +$('seek').value / 100 * buffer.duration;
  $('elapsed').textContent = format(offset);
  seeking = false;
  if (was) start();
};

$('gain').oninput = e => {
  gain = +e.target.value;
  $('gainval').textContent = gain.toFixed(1) + '×';
};

$('smooth').oninput = e => {
  smoothing = +e.target.value / 100;
  $('smoothval').textContent = e.target.value + '%';
};

$('volume').oninput = e => {
  $('volumeval').textContent = e.target.value + '%';
  if (master && mode !== 'mic')
    master.gain.setTargetAtTime(+e.target.value / 100, ctx.currentTime, 0.03);
};

/* 拖拽旋转 */
canvas.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', e => {
  if (!drag) return;

  const rect = canvas.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height * 0.55;   // 与 project() 的 0.55 对齐

  // 指针在画布中心"上方"→ 后半（far）；"下方"→ 前半（near）
  const isFront = e.clientY >= cy;

  const dx = e.clientX - drag.x;
  // 前半区：向右拖 → 角度减小；后半区：向右拖 → 角度增大（方向相反）
  angle += (isFront ? -1 : 1) * dx * 0.006;

  tilt = Math.max(0.3, Math.min(0.8, tilt + (e.clientY - drag.y) * 0.005));
  drag = { x: e.clientX, y: e.clientY };
  staticDirty = true;   // ← 保留：角度变了要重建静态几何 + 排序
});

canvas.addEventListener('pointerup', () => drag = null);
canvas.addEventListener('pointercancel', () => drag = null);

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.5, Math.min(2.2, zoom - e.deltaY * 0.001));
  staticDirty = true;
}, { passive: false });

$('reset').onclick = () => {
  angle = -0.32;
  tilt = 0.58;
  zoom = 1;
  staticDirty = true;
};

$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (e) {
    $('message').textContent = '当前浏览器不支持全屏。';
  }
};

document.addEventListener('fullscreenchange', () => {
  $('fullscreen').textContent = document.fullscreenElement ? '退出全屏 ↙' : '全屏 ↗';
});

window.addEventListener('pagehide', stop);

/* ============================================================
 *  启动
 * ============================================================ */
size();
requestAnimationFrame(frame);

$('ssaa').textContent = Bloom.ssaa + 'x';
$('bloomLevels').textContent = Bloom.levels + 'x';
update();

/* ============================================================
 *  调试钩子（修复项 9：只建一次）
 * ============================================================ */
window.__dbg = {
  gl, Bloom, prog, staticVBO, dynVBO, A,
  get staticCount() { return staticCount; },
  get dynCount()    { return dynCount; },
  ext: gl.getExtension('EXT_disjoint_timer_query_webgl2')
};