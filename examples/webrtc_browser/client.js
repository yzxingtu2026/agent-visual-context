"use strict";
const $ = id => document.getElementById(id);
let peer = null, stream = null, channel = null, timer = null, previousVideo = null;
let running = false, generation = 0, pingSent = null, latest = null, latestAt = 0, durations = [];
const status = text => { $("status").textContent = text; };

function stop() {
  generation++;
  running = false;
  clearInterval(timer); timer = null;
  if (channel) channel.close();
  if (peer) peer.close();
  if (stream) stream.getTracks().forEach(track => track.stop());
  channel = peer = stream = null;
  $("local").srcObject = $("remote").srcObject = null;
  $("start").disabled = false; $("stop").disabled = true;
  $("fps").disabled = false;
  latest = previousVideo = pingSent = null;
  status("已停止，摄像头已释放");
}

function gatherIce(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { cleanup(); reject(new Error("ICE 候选收集超时")); }, 10000);
    function cleanup() { clearTimeout(timeout); pc.removeEventListener("icegatheringstatechange", changed); }
    function changed() { if (pc.iceGatheringState === "complete") { cleanup(); resolve(); } }
    pc.addEventListener("icegatheringstatechange", changed);
  });
}

function percentile(values, p) {
  const ordered = [...values].sort((a,b) => a-b);
  return ordered[Math.max(0, Math.ceil(ordered.length*p)-1)].toFixed(1);
}

async function stats(pc) {
  if (pc !== peer || pc.connectionState !== "connected") return;
  try {
    const report = await pc.getStats();
    if (pc !== peer) return;
    report.forEach(s => {
      if (s.type === "inbound-rtp" && s.kind === "video") {
        let fps = s.framesPerSecond;
        if (fps == null && previousVideo && s.timestamp > previousVideo.timestamp) {
          fps = 1000*(s.framesDecoded-previousVideo.framesDecoded)/(s.timestamp-previousVideo.timestamp);
        }
        $("videoFps").textContent = fps == null ? "—" : `${fps.toFixed(1)} FPS`;
        previousVideo = s;
      }
    });
    if (channel?.readyState === "open" && (pingSent == null || performance.now()-pingSent > 5000)) {
      pingSent = performance.now(); channel.send("ping");
    }
  } catch (error) { if (pc === peer) status(`统计失败：${error.message}`); }
}

async function start() {
  $("start").disabled = true; $("stop").disabled = false; $("fps").disabled = true;
  running = true;
  const run = ++generation;
  durations = []; latest = null; previousVideo = null;
  ["videoFps", "rtt", "inference", "dropped"].forEach(id => { $(id).textContent = "—"; });
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("摄像头不可用，请通过 http://localhost:8080 打开");
    status("等待摄像头授权…");
    const acquired = await navigator.mediaDevices.getUserMedia({audio:false, video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:Number($("fps").value)}}});
    if (run !== generation) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; $("local").srcObject = stream;
    const pc = peer = new RTCPeerConnection({iceServers:[]});
    pc.ontrack = event => { if (pc === peer) $("remote").srcObject = new MediaStream([event.track]); };
    pc.onconnectionstatechange = () => {
      if (pc !== peer) return;
      status(`连接状态：${pc.connectionState}`);
      if (pc.connectionState === "failed") { stop(); status("连接失败，请重试并检查浏览器 WebRTC 支持"); }
    };
    const dc = channel = pc.createDataChannel("results", {ordered:false, maxRetransmits:0});
    dc.onmessage = event => {
      if (pc !== peer) return;
      if (event.data === "pong") {
        if (pingSent != null) $("rtt").textContent = `${(performance.now()-pingSent).toFixed(1)} ms`;
        pingSent = null; return;
      }
      try {
        const data = JSON.parse(event.data);
        $("result").textContent = JSON.stringify(data, null, 2);
        if (data.type !== "inference") return;
        latest = data; latestAt = performance.now();
        durations.push(data.server_ms); if (durations.length > 300) durations.shift();
        $("inference").textContent = `${percentile(durations,.5)} / ${percentile(durations,.95)} ms`;
        $("dropped").textContent = `${data.queue_dropped} / ${data.processed}`;
        $("backend").textContent = data.backend;
      } catch (error) { status(`结果解析失败：${error.message}`); }
    };
    stream.getTracks().forEach(track => pc.addTrack(track, stream));
    status("协商 WebRTC…");
    await pc.setLocalDescription(await pc.createOffer());
    await gatherIce(pc);
    if (run !== generation) return;
    const response = await fetch("/offer", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(pc.localDescription), signal:AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error(await response.text());
    const answer = await response.json();
    if (run !== generation) return;
    await pc.setRemoteDescription(answer);
    timer = setInterval(() => stats(pc), 1000);
  } catch (error) {
    if (run !== generation) return;
    stop(); status(`启动失败：${error.message}`);
  }
}

function draw() {
  const canvas = $("overlay"), video = $("remote");
  const width = video.videoWidth || 640, height = video.videoHeight || 480;
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,width,height);
  if (running && latest && performance.now()-latestAt < 2000) {
    ctx.strokeStyle = "#6cffa9"; ctx.fillStyle = "#6cffa9"; ctx.lineWidth = 2; ctx.font = "15px sans-serif";
    for (const item of latest.detections || []) {
      const b = item.bbox;
      ctx.strokeRect(b.x,b.y,b.width,b.height);
      ctx.fillText(`${item.label} ${(item.confidence*100).toFixed(0)}%`,b.x,Math.max(16,b.y-5));
    }
  }
  requestAnimationFrame(draw);
}
$("start").onclick = start; $("stop").onclick = stop;
window.addEventListener("pagehide", stop);
draw();
