# Reactor Audio Visualizer

以切尔诺贝利反应堆堆芯的控制棒阵列为视觉原型的实时音频可视化实验。

网页使用 Web Audio API 在浏览器本机分析音频，使用WebGL2将不同频段的能量渲染为 709 根控制棒的升降运动。支持演示信号、本地音频文件、麦克风输入、视角拖动与缩放，以及灵敏度、平滑度、全局缩放、Bloom泛光和音量调节等。

## 本地运行

```bash
python3 -m http.server 4173 --directory dist
```

Then open <http://localhost:4173>.

## 注意事项

这是一个互动实验。非真实物理模拟，音频仅在本机处理，不会向任何服务器上传任何内容。

## 许可

MIT. 详见 [LICENSE](LICENSE).
