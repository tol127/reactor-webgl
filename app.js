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
let smoothPeakIndex = 0;
let seeking = false;
let fpsLast = 0, fpsFrames = 0;
const FFTSize = 2048;
// ★★★ 开关：true = 多级，false = 单级 ★★★
const MultiBloom = false;
const FBO_SS = 1;   //SSAA
const FBO_MS = 4;   //MSAA

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

// WebGL2 的实例化是核心功能，ANGLE_instanced_arrays 扩展已不再需要

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
 *  Bloom 后处理模块（含 SSAA + MSAA）
 * ============================================================ */
const Bloom = (function () {

  let gl = null;
  let width = 1, height = 1;

  // ★★★ 开关：true = 多级，false = 单级 ★★★
  const MULTI_LEVEL = MultiBloom;

  // 多级用
  const LEVELS = 4;
  let bloomFBOs = [];
  let blurFBOs = [];

  // 单级用
  let fboBright = null, fboBlur = null;

  // 场景 FBO（SSAA 后的普通纹理，也是 MSAA resolve 目标）
  let fboScene = null;
  // 场景 MSAA FBO（多重采样渲染目标）
  let fboSceneMS = null;

  let progBright = null, progBlur = null;
  let progCompositeMulti = null;
  let progCompositeSingle = null;
  let quadBuf = null;

  let threshold = 0.45;
  let intensity = 1.6;
  let radius = 2.5;
  let sceneHasAlpha = true;

  let msaaEnabled = false;
  let msaaSamples = 1;

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

  // 多级合成
  const FS_COMPOSITE_MULTI = `#version 300 es
    precision mediump float;
    in vec2 vUV;
    uniform sampler2D uScene;
    uniform sampler2D uBloom0;
    uniform sampler2D uBloom1;
    uniform sampler2D uBloom2;
    uniform sampler2D uBloom3;
    uniform float uIntensity;
    out vec4 fragColor;
    void main(){
      vec4 scene = texture(uScene, vUV);
      // 权重和 > 1，叠加而非平均
      vec3 b = texture(uBloom0, vUV).rgb * 0.9
             + texture(uBloom1, vUV).rgb * 0.5
             + texture(uBloom2, vUV).rgb * 0.3
             + texture(uBloom3, vUV).rgb * 0.15;
      b *= scene.a;
      vec3 color = scene.rgb + b * uIntensity;
      // 轻量 tone mapping
      float maxC = max(max(color.r, color.g), color.b);
      color *= (1.0 - smoothstep(1.0, 1.6, maxC) * 0.4);
      fragColor = vec4(color, scene.a);
    }`;

  // 单级合成
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

  /* ---------- 普通纹理 FBO ---------- */

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

  /* ---------- 多重采样 FBO（MSAA） ---------- */

  function createMSFBO(w, h, samples) {
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

    const colorRB = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, colorRB);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colorRB);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('MS FBO incomplete: 0x' + status.toString(16));

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    return { fbo, colorRB, w, h, samples };
  }

  function destroyMSFBO(f) {
    if (!f) return;
    gl.deleteFramebuffer(f.fbo);
    gl.deleteRenderbuffer(f.colorRB);
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

  /* ---------- 多级模糊辅助 ---------- */

  function blurLevel(i, r) {
    const src = bloomFBOs[i];
    const tmp = blurFBOs[i];

    gl.bindFramebuffer(gl.FRAMEBUFFER, tmp.fbo);
    gl.viewport(0, 0, tmp.w, tmp.h);
    gl.useProgram(progBlur);
    bindTexture(src.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBlur, 'uTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'uDirection'), r / tmp.w, 0);
    drawQuad(progBlur);

    gl.bindFramebuffer(gl.FRAMEBUFFER, src.fbo);
    gl.viewport(0, 0, src.w, src.h);
    bindTexture(tmp.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBlur, 'uTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'uDirection'), 0, r / src.h);
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
    destroyMSFBO(fboSceneMS);
    fboSceneMS = null;
    destroyFBO(fboBright);
    destroyFBO(fboBlur);
    bloomFBOs.forEach(destroyFBO);
    blurFBOs.forEach(destroyFBO);
    bloomFBOs = [];
    blurFBOs = [];

    // 场景 SSAA 尺寸
    const SS = FBO_SS;
    const sw = Math.max(1, Math.round(width * SS));
    const sh = Math.max(1, Math.round(height * SS));

    // SSAA 目标纹理（resolve 后作为 Bloom 输入）
    fboScene = createFBO(sw, sh);

    // MSAA 实际渲染目标（与 fboScene 同尺寸，1:1 blit）
    msaaSamples = Math.max(1, Math.min(FBO_MS, gl.getParameter(gl.MAX_SAMPLES)));
    msaaEnabled = msaaSamples > 1;
    if (msaaEnabled) {
      fboSceneMS = createMSFBO(sw, sh, msaaSamples);
    }

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
    const target = msaaEnabled ? fboSceneMS : fboScene;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, target.w, target.h);
    if (sceneHasAlpha) {
      gl.clearColor(0, 0, 0, 0);
    } else {
      gl.clearColor(0.071, 0.086, 0.078, 1);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  function endScene() {
    // MSAA resolve → fboScene
    if (msaaEnabled) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fboSceneMS.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fboScene.fbo);
      gl.blitFramebuffer(
        0, 0, fboSceneMS.w, fboSceneMS.h,
        0, 0, fboScene.w, fboScene.h,
        gl.COLOR_BUFFER_BIT, gl.NEAREST
      );
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    }

    if (MULTI_LEVEL) {
      endSceneMulti();
    } else {
      endSceneSingle();
    }
  }

  /* ---- 多级 ---- */
  function endSceneMulti() {
    // Pass 1: 亮度提取 → bloom0
    const b0 = bloomFBOs[0];
    gl.bindFramebuffer(gl.FRAMEBUFFER, b0.fbo);
    gl.viewport(0, 0, b0.w, b0.h);
    gl.useProgram(progBright);
    bindTexture(fboScene.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBright, 'uScene'), 0);
    gl.uniform1f(gl.getUniformLocation(progBright, 'uThreshold'), threshold);
    drawQuad(progBright);

    // Pass 2: 逐级
    for (let i = 0; i < LEVELS; i++) {
      if (i > 0) {
        const cur = bloomFBOs[i];
        const prev = bloomFBOs[i - 1];
        gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fbo);
        gl.viewport(0, 0, cur.w, cur.h);
        gl.useProgram(progBlur);
        bindTexture(prev.tex, 0);
        gl.uniform1i(gl.getUniformLocation(progBlur, 'uTexture'), 0);
        gl.uniform2f(gl.getUniformLocation(progBlur, 'uDirection'), 0, 0);
        drawQuad(progBlur);
      }
      blurLevel(i, radius);
    }

    // Pass 3: 合成
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(progCompositeMulti);
    bindTexture(fboScene.tex, 0);
    bindTexture(bloomFBOs[0].tex, 1);
    bindTexture(bloomFBOs[1].tex, 2);
    bindTexture(bloomFBOs[2].tex, 3);
    bindTexture(bloomFBOs[3].tex, 4);
    gl.uniform1i(gl.getUniformLocation(progCompositeMulti, 'uScene'), 0);
    gl.uniform1i(gl.getUniformLocation(progCompositeMulti, 'uBloom0'), 1);
    gl.uniform1i(gl.getUniformLocation(progCompositeMulti, 'uBloom1'), 2);
    gl.uniform1i(gl.getUniformLocation(progCompositeMulti, 'uBloom2'), 3);
    gl.uniform1i(gl.getUniformLocation(progCompositeMulti, 'uBloom3'), 4);
    gl.uniform1f(gl.getUniformLocation(progCompositeMulti, 'uIntensity'), intensity);
    drawQuad(progCompositeMulti);
  }

  /* ---- 单级 ---- */
  function endSceneSingle() {
    const bw = fboBright.w, bh = fboBright.h;

    // Pass 1: 亮度提取
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBright.fbo);
    gl.viewport(0, 0, bw, bh);
    gl.useProgram(progBright);
    bindTexture(fboScene.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBright, 'uScene'), 0);
    gl.uniform1f(gl.getUniformLocation(progBright, 'uThreshold'), threshold);
    drawQuad(progBright);

    // Pass 2: 水平
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBlur.fbo);
    gl.viewport(0, 0, bw, bh);
    gl.useProgram(progBlur);
    bindTexture(fboBright.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBlur, 'uTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'uDirection'), radius / bw, 0);
    drawQuad(progBlur);

    // Pass 3: 垂直
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboBright.fbo);
    gl.viewport(0, 0, bw, bh);
    bindTexture(fboBlur.tex, 0);
    gl.uniform1i(gl.getUniformLocation(progBlur, 'uTexture'), 0);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'uDirection'), 0, radius / bh);
    drawQuad(progBlur);

    // Pass 4: 合成
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.useProgram(progCompositeSingle);
    bindTexture(fboScene.tex, 0);
    bindTexture(fboBright.tex, 1);
    gl.uniform1i(gl.getUniformLocation(progCompositeSingle, 'uScene'), 0);
    gl.uniform1i(gl.getUniformLocation(progCompositeSingle, 'uBloom'), 1);
    gl.uniform1f(gl.getUniformLocation(progCompositeSingle, 'uIntensity'), intensity);
    drawQuad(progCompositeSingle);
  }

  function setParams(p) {
    if (p.threshold !== undefined) threshold = p.threshold;
    if (p.intensity !== undefined) intensity = p.intensity;
    if (p.radius !== undefined) radius = p.radius;
  }

  return { 
    init, resize, beginScene, endScene, setParams,
    //get mode()  { return MULTI_LEVEL ? 'Multi' : 'Single'; },
    get levels(){ return MULTI_LEVEL ? LEVELS : 1; },
    get ssaa()  { return FBO_SS; },
    get msaa()  { return msaaSamples; } 
  };
  
})();

/* ============================================================
 *  顶点缓冲
 * ============================================================ */
const MAX_VERTS = 160000;
const vertexData = new Float32Array(MAX_VERTS * 6);
let vertexCount = 0;

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, vertexData.byteLength, gl.DYNAMIC_DRAW);

/* ============================================================
 *  等距投影
 * ============================================================ */
let canvasW = 1, canvasH = 1;

function project(x, z, h) {
  const s = Math.min(canvasW / 45, canvasH / 37) * zoom;
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const rx = x * ca - z * sa;
  const rz = x * sa + z * ca;
  return [
    canvasW / 2 + rx * s,
    canvasH * 0.55 + rz * s * tilt - h * s
  ];
}

function toClip(px, py) {
  return [
    (px / canvasW) * 2 - 1,
    1 - (py / canvasH) * 2
  ];
}

/* ============================================================
 *  颜色工具
 * ============================================================ */
function hex2rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
function rgb(r, g, b) {
  return [r / 255, g / 255, b / 255];
}

const COL = {
  ringOuterDark: hex2rgb('#060b07'),
  ringMain:      hex2rgb('#505948'),
  ringInner:     hex2rgb('#878969'),
  ringOuter:     hex2rgb('#303a2d'),
  tickA:         hex2rgb('#484d31'),
  tickB:         hex2rgb('#adac64'),
  side1:         hex2rgb('#424838'),
  side3:         hex2rgb('#333c2b'),
  side4:         hex2rgb('#39432f'),
  sideEdge:      hex2rgb('#111910'),
  topEdge:       hex2rgb('#232d20'),
  rodDot:        hex2rgb('#404735'),
  edge:          hex2rgb('#0d1409')
};

/* ============================================================
 *  几何生成
 * ============================================================ */
let vp = 0;

function pushVertex(px, py, col) {
  const c = toClip(px, py);
  vertexData[vp++] = c[0];
  vertexData[vp++] = c[1];
  vertexData[vp++] = 0;
  vertexData[vp++] = col[0];
  vertexData[vp++] = col[1];
  vertexData[vp++] = col[2];
}

function pushEdge(p1, p2, col, EW) {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len * EW * 0.5;
  const ny =  dx / len * EW * 0.5;
  const A = [p1[0] + nx, p1[1] + ny];
  const B = [p2[0] + nx, p2[1] + ny];
  const C = [p2[0] - nx, p2[1] - ny];
  const D = [p1[0] - nx, p1[1] - ny];
  pushVertex(A[0], A[1], col);
  pushVertex(B[0], B[1], col);
  pushVertex(C[0], C[1], col);
  pushVertex(A[0], A[1], col);
  pushVertex(C[0], C[1], col);
  pushVertex(D[0], D[1], col);
}

function pushQuad(p1, p2, p3, p4, col) {
  pushVertex(p1[0], p1[1], col);
  pushVertex(p2[0], p2[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p1[0], p1[1], col);
  pushVertex(p3[0], p3[1], col);
  pushVertex(p4[0], p4[1], col);
}

// 圆盘近似（用三角扇形），用于精细圆点
function pushDisc(cx, cy, rx, ry, col, SEG) {
  SEG = SEG || 16;
  for (let i = 0; i < SEG; i++) {
    const a0 = i / SEG * Math.PI * 2;
    const a1 = (i + 1) / SEG * Math.PI * 2;
    pushVertex(cx, cy, col);
    pushVertex(cx + Math.cos(a0) * rx, cy + Math.sin(a0) * ry, col);
    pushVertex(cx + Math.cos(a1) * rx, cy + Math.sin(a1) * ry, col);
  }
}

/* ============================================================
 *  尺寸 & Bloom 初始化
 * ============================================================ */
function size() {
  const rect = canvas.getBoundingClientRect();
  const d = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * d));
  canvas.height = Math.max(1, Math.round(rect.height * d));
  canvasW = canvas.clientWidth;
  canvasH = canvas.clientHeight;
  gl.viewport(0, 0, canvas.width, canvas.height);

  // Bloom 尺寸同步
  Bloom.resize(canvas.width, canvas.height);

  const srect = spec.getBoundingClientRect();
  spec.width = Math.max(1, Math.round(srect.width * d));
  spec.height = Math.max(1, Math.round(srect.height * d));
  sg.setTransform(d, 0, 0, d, 0, 0);
}

// 先设初始尺寸，再初始化 Bloom（Bloom.init 内部会调 resize）
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
  if (playing && bins) {
    for (let i = 2; i < bins.length - 2; i++) {
      if (bins[i] > peak) { peak = bins[i]; peakIndex = i; }
    }
    smoothPeakIndex = smoothPeakIndex * 0.7 + peakIndex * 0.3;
  }
  energy = 0;
  if (playing && wave) {
    analyser.getFloatTimeDomainData(wave);
    for (let i = 0; i < wave.length; i++) energy += wave[i] * wave[i];
    energy = Math.sqrt(energy / wave.length);
  }

  // ---- Bloom 动态参数（和音频能量联动） ----
  const bloomIntensity = 1.2 + energy * 0.25;
  const bloomThreshold = 0.4 - energy * 0.15;
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

  // ---- 写顶点 ----
  vp = 0;

  // 1. 外圈环
  function drawRing(radius, colorHex, widthPx) {
    const col = hex2rgb(colorHex);
    const s = Math.min(canvasW / 45, canvasH / 37) * zoom;
    const halfW = (widthPx * 0.5) / s;
    const step = 96;
    for (let i = 0; i < step; i++) {
      const a0 = i / step * Math.PI * 2;
      const a1 = (i + 1) / step * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);
      const rIn = radius - halfW;
      const rOut = radius + halfW;
      const p0i = project(c0 * rIn, s0 * rIn, 0);
      const p1i = project(c1 * rIn, s1 * rIn, 0);
      const p0o = project(c0 * rOut, s0 * rOut, 0);
      const p1o = project(c1 * rOut, s1 * rOut, 0);
      pushQuad(p0i, p1i, p1o, p0o, col);
    }
  }

  drawRing(16.4, '#060b07', 60 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(16.5, '#505948',  6 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(16.0, '#878969',  3 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);
  drawRing(17.3, '#303a2d',  3 * (Math.min(canvasW / 45, canvasH / 37) * zoom) * 0.02);

  // 2. 80 刻度块
  for (let i = 0; i < 80; i++) {
    const a = i / 80 * Math.PI * 2;
    const a2 = a + 0.034;
    const r0 = 15.7, r1 = 16.15;
    const p1 = project(Math.cos(a) * r0, Math.sin(a) * r0, 0);
    const p2 = project(Math.cos(a2) * r0, Math.sin(a2) * r0, 0);
    const p3 = project(Math.cos(a2) * r1, Math.sin(a2) * r1, 0);
    const p4 = project(Math.cos(a) * r1, Math.sin(a) * r1, 0);
    const col = (i % 2 === 1) ? COL.tickA : COL.tickB;
    pushQuad(p1, p2, p3, p4, col);
  }

  // 3. 柱子（深度排序）
  const sorted = [...rods].sort((a, b) =>
    (a.x * Math.sin(angle) + a.z * Math.cos(angle)) -
    (b.x * Math.sin(angle) + b.z * Math.cos(angle))
  );

  const S = 0.44;
  const dotSize = Math.min(canvasW / 45, canvasH / 37) * zoom * 0.12;

  for (const rod of sorted) {
    const amp = rod._amp || 0;
    const h = rod.h;
    const x = rod.x, z = rod.z;

    const a0 = project(x - S, z - S, 0);
    const b0 = project(x + S, z - S, 0);
    const c0 = project(x + S, z + S, 0);
    const d0 = project(x - S, z + S, 0);
    const a  = project(x - S, z - S, h);
    const b  = project(x + S, z - S, h);
    const c  = project(x + S, z + S, h);
    const d  = project(x - S, z + S, h);

    pushQuad(a, b, b0, a0, COL.side1);
    pushQuad(b, c, c0, b0, rgb(55 + amp * 30, 62 + amp * 30, 44 + amp * 20));
    pushQuad(c, d, d0, c0, COL.side3);
    pushQuad(d, a, a0, d0, COL.side4);

    const s = rod.seed;
    let base;
    if (s > 0.92) base = [143, 109, 81];
    else if (s > 0.85) base = [97, 118, 104];
    else base = [139 + s * 29, 148 + s * 22, 119 + s * 21];
    pushQuad(a, b, c, d, rgb(
      base[0] + amp * 43,
      base[1] + amp * 40,
      base[2] + amp * 24
    ));

    // 顶面 4 边
    const EW = Math.max(0.8 * zoom, 1);
    const EC = COL.edge;
    pushEdge(a, b, EC, EW);
    pushEdge(b, c, EC, EW);
    pushEdge(c, d, EC, EW);
    pushEdge(d, a, EC, EW);

    // 朝镜头的两根竖棱
    const sa = Math.sin(angle), ca = Math.cos(angle);
    const rzA = -sa - ca, rzB = sa - ca, rzC = sa + ca, rzD = -sa + ca;
    const arr = [
      { rz: rzA, p1: a, p2: a0 },
      { rz: rzB, p1: b, p2: b0 },
      { rz: rzC, p1: c, p2: c0 },
      { rz: rzD, p1: d, p2: d0 }
    ];
    arr.sort((m, n) => n.rz - m.rz);
    pushEdge(arr[0].p1, arr[0].p2, EC, EW);
    pushEdge(arr[1].p1, arr[1].p2, EC, EW);

    // 圆点（精细圆盘）
    const cp = project(x, z, h + 0.01);
    pushDisc(cp[0], cp[1], dotSize, dotSize * tilt, COL.rodDot, 12);
  }

  vertexCount = vp / 6;

  /* ==========================================================
   *  渲染：Bloom 包裹
   * ========================================================== */
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertexData.subarray(0, vp));

  gl.disable(gl.DEPTH_TEST);

  Bloom.beginScene();

  // 场景绘制
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.enableVertexAttribArray(A.pos);
  gl.vertexAttribPointer(A.pos, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(A.color);
  gl.vertexAttribPointer(A.color, 3, gl.FLOAT, false, 24, 12);
  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);

  Bloom.endScene();

  /* ==========================================================
   *  频谱小图（Canvas 2D，独立于 WebGL）
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
    $('freq').innerHTML = (playing && peak > 0 ? Math.round(smoothPeakIndex * binHz) : '—') + '<small>Hz</small>';
    if (playing && !seeking) {
      const seconds = offset + ctx.currentTime - started;
      $('elapsed').textContent = format(seconds);
      if (buffer && mode === 'file') $('seek').value = seconds / buffer.duration * 100;
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
      const id = ++session;
      loading = true;
      $('message').textContent = '等待麦克风授权…';
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (id !== session) {
        s.getTracks().forEach(t => t.stop());
        return;
      }
      stream = s;
      source = ctx.createMediaStreamSource(s);
      source.connect(analyser);
      master.gain.value = 0;
      $('message').textContent = '正在监听麦克风；停止或切换输入即可关闭。';
    } else {
      master.gain.value = +$('volume').value / 100;

      if (mode === 'file' && buffer) {
        if (offset >= buffer.duration) offset = 0;
        source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(analyser);
        source.onended = () => {
          stop();
          offset = 0;
          $('seek').value = 0;
          $('elapsed').textContent = '00:00';
        };
        source.start(0, offset);
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
    stop();
    $('message').textContent = mode === 'mic'
      ? '无法访问麦克风，请检查浏览器权限，或导入音频。'
      : '无法播放此音频，请尝试其他文件。';
  } finally {
    if (attempt === session || !playing) loading = false;
  }
}

/* ============================================================
 *  事件绑定
 * ============================================================ */
$('play').onclick = () => {
  if (playing) {
    offset += ctx.currentTime - started;
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
    $('message').textContent = '无法读取该文件，请尝试 MP3 或 WAV 音频。';
  } finally {
    loading = false;
    //e.target.value = '';
  }
};

$('seek').onkeydown = e => {
  if (mode !== 'file' || !buffer) return;
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
  if (mode !== 'file' || !buffer) return;
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
  angle -= (e.clientX - drag.x) * 0.006;
  tilt = Math.max(0.3, Math.min(0.8, tilt + (e.clientY - drag.y) * 0.005));
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', () => drag = null);
canvas.addEventListener('pointercancel', () => drag = null);

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.5, Math.min(2.2, zoom - e.deltaY * 0.001));
}, { passive: false });

$('reset').onclick = () => {
  angle = -0.32;
  tilt = 0.58;
  zoom = 1;
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
$('msaa').textContent = Bloom.msaa + 'x';
//$('bloomMode').textContent   = Bloom.mode;
$('bloomLevels').textContent = Bloom.levels + 'x';
//$('fftSize').textContent = FFTSize
update();


/* ============================================================
 *  可选：Model Context 工具注册
 * ============================================================ 
 *  这里暂时用不到
 
if (document.modelContext?.registerTool) {
  try {
    Promise.resolve(document.modelContext.registerTool({
      name: 'configure_reactor_response',
      description: 'Adjust the reactor audio visualizer sensitivity and motion smoothing.',
      inputSchema: {
        type: 'object',
        properties: {
          sensitivity: { type: 'number', minimum: 0.3, maximum: 3 },
          smoothing: { type: 'number', minimum: 0, maximum: 95 }
        },
        required: ['sensitivity', 'smoothing'],
        additionalProperties: false
      },
      execute(input) {
        if (!input ||
            !Number.isFinite(input.sensitivity) || input.sensitivity < 0.3 || input.sensitivity > 3 ||
            !Number.isFinite(input.smoothing) || input.smoothing < 0 || input.smoothing > 95)
          throw Error('Invalid response settings');
        $('gain').value = input.sensitivity;
        $('gain').dispatchEvent(new Event('input'));
        $('smooth').value = input.smoothing;
        $('smooth').dispatchEvent(new Event('input'));
        return { sensitivity: gain, smoothing: smoothing * 100 };
      }
    })).catch(() => {});
  } catch (e) {}
}
*/