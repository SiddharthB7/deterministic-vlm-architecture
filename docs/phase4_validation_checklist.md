# Phase 4 Validation Checklist

- Verify the color topic exists: `ros2 topic echo /camera/color/image_raw`
- Verify Kurat receives frames: watch `/kurat/status` for `frame_age_s`
- Verify chat query works without camera dependence
- Verify a scene query uses the latest frame
- Verify a simple find query uses YOLO only
- Verify a rich find query uses YOLO plus Moondream
- Verify stale frame handling returns a clear reply instead of crashing
- Verify Ollama is reachable on the configured host
- Verify `qwen2.5:3b-instruct` is pulled
- Verify `moondream` is pulled
