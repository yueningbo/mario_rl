"""Local web page that plots training curves from ``artifacts/*/train.log``.

Read-only: it never touches a running training process. Uses only the standard library.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LINE = re.compile(
    r"update (\d+)/(\d+) steps (\d+) .*?\| mean_x (\d+) best_x (\d+) wins\(last40\) (\d+)"
)


def read_runs(root: Path) -> dict[str, list[dict[str, int]]]:
    """Parse every ``<root>/*/train.log`` into a list of metric points."""
    runs: dict[str, list[dict[str, int]]] = {}
    for log in sorted(root.glob("*/train.log")):
        points = []
        for line in log.read_text(errors="replace").splitlines():
            match = LINE.search(line)
            if match:
                step, _, _, mean_x, best_x, wins = (int(v) for v in match.groups())
                points.append(
                    {
                        "steps": int(match.group(3)),
                        "mean_x": mean_x,
                        "best_x": best_x,
                        "wins": wins,
                        "update": step,
                    }
                )
        if points:
            runs[log.parent.name] = points
    return runs


PAGE = """<!doctype html><meta charset="utf-8"><title>Mario training</title>
<style>
body{font:14px system-ui;margin:20px;background:#111;color:#ddd}
h2{margin:18px 0 4px;font-size:15px} canvas{background:#1b1b1b;border:1px solid #333}
#legend span{margin-right:16px} #status{color:#888;margin-left:12px}
</style>
<h1 style="font-size:18px">Mario training <span id="status"></span></h1>
<div id="legend"></div>
<h2>Mean x reached (last 40 episodes) - flag is at x=3161</h2><canvas id="c1" width="900" height="260"></canvas>
<h2>Clears in last 40 episodes (noisy, small numbers)</h2><canvas id="c2" width="900" height="160"></canvas>
<script>
const COLORS=["#4fc3f7","#ffb74d","#81c784","#ba68c8","#e57373","#fff176"];
function draw(id,runs,key,maxY,flag){
  const c=document.getElementById(id),g=c.getContext("2d"),W=c.width,H=c.height,P=40;
  g.clearRect(0,0,W,H);
  let maxX=1;for(const r of Object.values(runs))for(const p of r)maxX=Math.max(maxX,p.steps);
  g.strokeStyle="#333";g.fillStyle="#888";g.font="11px system-ui";
  for(let i=0;i<=4;i++){const y=H-P/2-(H-P)*i/4;g.beginPath();g.moveTo(P,y);g.lineTo(W-8,y);g.stroke();
    g.fillText(Math.round(maxY*i/4),4,y+4);}
  for(let i=0;i<=4;i++)g.fillText((maxX*i/4/1e6).toFixed(2)+"M",P+(W-P-8)*i/4-14,H-4);
  if(flag){const y=H-P/2-(H-P)*flag/maxY;g.strokeStyle="#4caf50";g.setLineDash([4,4]);
    g.beginPath();g.moveTo(P,y);g.lineTo(W-8,y);g.stroke();g.setLineDash([]);}
  let i=0;for(const r of Object.values(runs)){
    g.strokeStyle=COLORS[i++%COLORS.length];g.lineWidth=1.6;g.beginPath();
    r.forEach((p,k)=>{const x=P+(W-P-8)*p.steps/maxX,y=H-P/2-(H-P)*p[key]/maxY;k?g.lineTo(x,y):g.moveTo(x,y)});
    g.stroke();}
}
async function tick(){
  try{
    const runs=await (await fetch("/data")).json(),names=Object.keys(runs);
    document.getElementById("legend").innerHTML=names.map((n,i)=>{
      const r=runs[n],l=r[r.length-1];
      return `<span style="color:${COLORS[i%COLORS.length]}">&#9632; ${n}: ${(l.steps/1e6).toFixed(2)}M steps, mean_x ${l.mean_x}, best ${l.best_x}</span>`}).join("");
    draw("c1",runs,"mean_x",3300,3161);draw("c2",runs,"wins",40,0);
    document.getElementById("status").textContent="updated "+new Date().toLocaleTimeString();
  }catch(e){document.getElementById("status").textContent="server unreachable"}
}
tick();setInterval(tick,5000);
</script>"""


def serve(root: Path, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/data":
                body, kind = json.dumps(read_runs(root)).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training curves: http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
