const byId = id => document.getElementById(id);
const canvas = byId("plot");
const series = [
  ["alpha", "#006b61"], ["eta", "#af4d20"], ["rho", "#67559e"]
];
let lastRun;

function draw() {
  if (!lastRun) return;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 55, top = 26, right = width - 12, bottom = height - 43;
  const log = byId("scale").value === "log";
  const values = series.flatMap(([name]) => lastRun[name]);
  const low = log ? Math.floor(Math.log10(Math.min(...values))) : 0;
  const high = log ? Math.ceil(Math.log10(Math.max(...values))) : Math.max(...values);
  const time = lastRun.t, end = time[time.length - 1];
  const sx = t => left + (right - left) * t / end;
  const sy = y => bottom - (bottom - top) * (y - low) / (high - low);
  const exponentText = n => n === 0 ? "1" : "1e" + n;
  ctx.font = "12px Arial, sans-serif";
  ctx.fillStyle = "#4b625c";
  ctx.fillText("Scaled concentration", left, 13);
  const ticks = log ? high - low : 4;
  for (let i = 0; i <= ticks; i++) {
    const value = low + i * (high - low) / ticks, py = sy(value);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py);
    ctx.strokeStyle = "#d8dfd5"; ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = "right";
    const label = log ? exponentText(value) : Number(value.toPrecision(2)).toLocaleString();
    ctx.fillText(label, left - 6, py + 4);
  }
  ctx.textAlign = "center";
  const divisions = width < 450 ? 3 : 6;
  for (let i = 0; i <= divisions; i++) {
    const tick = end * i / divisions;
    ctx.fillText(String(Number(tick.toFixed(2))), sx(tick), bottom + 17);
  }
  ctx.fillText("Model time", (left + right) / 2, height - 3);
  ctx.save(); ctx.beginPath(); ctx.rect(left, top - 1, right - left, bottom - top + 2); ctx.clip();
  for (const [name, colour] of series) {
    ctx.beginPath(); ctx.strokeStyle = colour; ctx.lineWidth = 2.2;
    lastRun[name].forEach((y, i) => {
      const x = sx(time[i]), py = sy(log ? Math.log10(y) : y);
      if (i === 0) ctx.moveTo(x, py); else ctx.lineTo(x, py);
    });
    ctx.stroke();
  }
  ctx.restore();
}

async function run() {
  const url = new URL("simulate/oregonator", document.baseURI);
  const body = {t_end: byId("end").valueAsNumber, f: byId("factor").valueAsNumber};
  byId("request").textContent = "POST " + url.pathname + "\nContent-Type: application/json\n\n" + JSON.stringify(body);
  byId("run").disabled = true;
  byId("status").textContent = "Computing…";
  try {
    const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (!response.ok) throw new Error("Request failed (HTTP " + response.status + "). Check the parameters and try again.");
    lastRun = await response.json();
    draw();
    byId("summary").textContent = "Model time 0–" + body.t_end + " · f = " + body.f + " · " + lastRun.t.length + " saved states";
    canvas.setAttribute("aria-label", "Scaled concentrations over model time 0 to " + body.t_end + ". Final values: " +
      series.map(([name]) => name + " " + lastRun[name].at(-1).toPrecision(3)).join(", "));
    byId("status").textContent = "";
  } catch (error) {
    byId("status").textContent = error.message || "Could not reach the simulation. Try again.";
  } finally {
    byId("run").disabled = false;
  }
}

byId("settings").addEventListener("submit", event => {event.preventDefault(); run();});
byId("scale").addEventListener("change", draw);
new ResizeObserver(draw).observe(canvas);
run();
