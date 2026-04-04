import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import time
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19

# ── Quadrant boundaries ───────────────────────────────────────────────────────
# The field is divided into four equal quadrants along both axes.
#
#   top_left  │  top_right
#   ──────────┼──────────
#   bot_left  │  bot_right
#
_MID_X = FIELD_WIDTH  / 2   # 0.79 m
_MID_Y = FIELD_HEIGHT / 2   # 1.095 m

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_master", broker=mb, print_every=100)

# ── Broker state ──────────────────────────────────────────────────────────────
_robot_pos    = None   # {"x": float, "y": float}
_other_robots = None   # {"robots": [...]} from prediction node
_ball         = None   # {"global_pos": {x,y}, "ball_lost": bool, ...}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quadrant(x, y):
    """Return the quadrant name for a global field position."""
    col = "right" if x >= _MID_X else "left"
    row = "top"   if y >= _MID_Y else "bottom"
    return f"{row}_{col}"


def _publish(now):
    state = {"t": round(now, 3)}

    if _robot_pos is not None:
        state["self"] = _quadrant(float(_robot_pos["x"]), float(_robot_pos["y"]))

    if _other_robots is not None:
        entries = []
        for r in _other_robots.get("robots", []):
            x, y = r.get("x"), r.get("y")
            if x is None or y is None:
                continue
            entries.append({
                "id":        r.get("id"),
                "quadrant":  _quadrant(float(x), float(y)),
                "predicted": r.get("method") == "predicted",
            })
        state["robots"] = entries

    if _ball is not None:
        gpos = _ball.get("global_pos")
        if gpos is not None:
            state["ball"] = {
                "quadrant": _quadrant(float(gpos["x"]), float(gpos["y"])),
                "lost":     bool(_ball.get("ball_lost", False)),
            }

    mb.set("field_sectors", json.dumps(state))


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _other_robots, _ball

    if value is None:
        return

    try:
        if key == "robot_position":
            _robot_pos = json.loads(value)
        elif key == "other_robots":
            _other_robots = json.loads(value)
        elif key == "ball":
            _ball = json.loads(value)
        else:
            return
    except (json.JSONDecodeError, TypeError):
        return

    with _perf.measure("sectors"):
        _publish(time.monotonic())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("robot_position", "other_robots", "ball"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["robot_position", "other_robots", "ball"], on_update)
    print("[MASTER] Starting master node...")
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[MASTER] Stopped.")
        mb.close()
