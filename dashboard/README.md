# Causal Repair Observatory

The dashboard suite serves the saved `Interaction Dominant` and `Ringing Trap`
trajectories directly from `runs/repair_showcases`. It requires only Python,
NumPy, and a modern browser.

From the repository root:

```powershell
python dashboard\server.py
```

The browser opens at `http://127.0.0.1:8765`. Use `--no-browser` for a server-only
launch or `--port 9000` to select another port.

Controls:

- choose either dashboard in the top navigation;
- play/pause or scrub the synchronized trajectory;
- click a spacetime heatmap to select a slice;
- switch the heatmap among `A_z`, active mismatch, repair energy, and `Phi`;
- use Space to play/pause and the arrow keys to step frames.

The API endpoints are `/api/health`, `/api/catalog`, and
`/api/showcase/<scenario>`.
