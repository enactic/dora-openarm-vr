# OpenArm VR Teleoperation

VR-based teleoperation for the OpenArm robot using Meta Quest 3 and dora-rs dataflows.

---

## VR Setup (Meta Quest 3)

### One-time setup

1. Create a [Meta Quest Developer account](https://developer.oculus.com/) and install [Meta Quest Developer Hub](https://developer.oculus.com/meta-quest-developer-hub/).
2. Download the teleoperation APK from the [link](https://drive.google.com/file/d/1lLDuoQAcl3YBKPE77F_1Z-8RzXEOFxpm/view?usp=drive_link).
3. Sideload the APK onto your Quest 3 via Developer Hub.

### Per-session setup

1. Put on the headset and launch the teleoperation app.
2. Press the **left controller menu button** — a settings panel will appear.
3. Enter the **IP address** of your PC host and the **port** (default: `5006`).
4. Verify communication from the PC host:
   ```bash
   nc -lu 5006
   ```
5. **Tape the center-of-eye sensor** on the headset to keep it constantly activated (prevents the display from sleeping when worn at the neck).
6. Take off the headset and hang it around your neck — you will operate with the controllers while the headset rests there.

## Safety Notes

- When launching the real robot dataflow, **gently pull the trigger slowly** at first to align the robot before making any larger movements.
- Always verify motion in simulation before running on hardware. -->

## Quick Start

Test running teleoperation in mujoco.

```bash
uv run dora build config/dataflow-mujoco.yaml --uv
uv run dora run config/dataflow-mujoco.yaml --uv
```